#!/usr/bin/env python3

import sys
import glob
import json
import time
import base64
import bisect
import jinja2
import random
import hashlib
import pathlib
import http.cookies
import peewee as pw
import urllib.parse
import chimedb.core as db
from datetime import datetime
from chimedb.core.mediawiki import MediaWikiUser
from chimedb.dataflag import (
    DataFlagClient,
    DataFlagOpinion,
    DataFlagOpinionType,
    DataRevision,
)

# The version string used to look up the client record in the DB.
_DB_VERSION = "v0.1"

# Session expiry time, in days.  May be fractional
SESSION_EXPIRY_DAYS = 7

# Header to invalidate the session cookie
INVALIDATE_SESSION = ("Set-Cookie", "dv_session=; SameSite=Strict; Secure; Max-Age=0;")

# Available pipeline revisions
REVISIONS = list()

# The DataFlagOpinionType and DataFlagClient we're using.  These are set on first use
OPINION_TYPE = None
CLIENT = None

# Directories
template_dir = pathlib.Path(__file__).with_name("templates")
render_dir = pathlib.Path(__file__).with_name("rendered")

# Set up Jinja2
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(template_dir), autoescape=jinja2.select_autoescape()
)


def set_globals():
    """Read globals from DB, if necessary."""

    global OPINION_TYPE, CLIENT
    if OPINION_TYPE is None:
        OPINION_TYPE = DataFlagOpinionType.get(name="bondia")
    if CLIENT is None:
        CLIENT = DataFlagClient.get(
            client_name="daily_viewer",
            client_version=_DB_VERSION,
        )

    # Find all available revisions
    if not REVISIONS:
        for datarev in DataRevision.select():
            # DataRevision.name is of the form rev_##; find the integer part
            try:
                revision = int(datarev.name[4:])
            except ValueError:
                # Skip datarevs with unparsable names
                continue

            # Now look for stuff
            for path in glob.iglob(f"rev{revision:02d}_????.html", root_dir=render_dir):
                REVISIONS.append(revision)
                break

        if REVISIONS:
            print(f"Available revisions: {REVISIONS!r}")
        else:
            raise RuntimeError(f"Nothing available in {render_dir}!")


def render_template(name, data, headers=None):
    """Render a Jinja2 template `name` using `data`."""

    template = jinja_env.get_template(name + ".html.j2")
    return 200, headers, "text/html; charset=utf-8", template.render(**data)


def encode_session(user, issue_time=None):
    """Given a user record, create a new session cookie."""

    # The session cookie is composed of three fields, separated by colons:
    # UID of user (=user.user_id)
    # C-time when the session was created
    # base64-encoded SHA224-hash of the first two fields along with
    #     user.user_name and user.user_password

    if issue_time is None:
        issue_time = int(time.time())

    # The data to hash
    text = f"{user.user_id}-{user.user_name}-{issue_time}-{user.user_password}"
    digest = base64.b64encode(
        hashlib.sha224(text.encode("utf-8")).digest(), altchars=b"-~"
    )

    return f"{user.user_id}:{issue_time}:{digest.decode('utf-8')}"


def decode_session(session):
    """Verify and decode `session`.  Returns None on error/failure."""

    parts = session.split(":")
    if len(parts) != 3:
        return None

    # Check for session expiry
    try:
        if int(parts[1]) + SESSION_EXPIRY_DAYS * 86400 <= time.time():
            return None
    except ValueError:
        # Non-numeric time
        return None

    # get the user record
    try:
        user = MediaWikiUser.get(user_id=parts[0])
    except pw.DoesNotExist:
        return None

    # Re-encode the session to see if it matches
    if encode_session(user, issue_time=parts[1]) != session:
        # Bad session value
        return None

    # Session verification succeeded
    return user


def check_login(query):
    """Attempt to authenticate.

    Returns
    -------
    success : bool
        True if login succeeded
    result : session or error
        On success, this is the user and cookie
        On failure, this is the error message
    """

    try:
        # Returns a 2-tuple: (user_name, user_id)
        result = MediaWikiUser.authenticate(query["user_name"], query["user_password"])
    except UserWarning as error:
        return False, str(error)

    # Get the record
    user = MediaWikiUser.get(user_id=result[1])

    # Authentication succeeded.  Create a new session cookie
    return True, (
        user,
        ("Set-Cookie", f"dv_session={encode_session(user)}; SameSite=Strict; Secure"),
    )


def strip_query(query):
    """Strip dict `query` of all unnecessary keys.

    Returns
    -------
    new_query: dict
        A new dict is returns with only the keys
        csd, rev, and fortnight (if present in `query`).
    """
    new_query = dict()

    if "csd" in query and query["csd"]:
        new_query["csd"] = query["csd"]
    if "rev" in query and query["rev"]:
        new_query["rev"] = query["rev"]
    if "fortnight" in query:
        new_query["fortnight"] = "yes"

    return new_query


def do_login(query):
    """Ask for, and perform login.

    On successful login, returns a tuple (True, user, cookie).
    On failure returns a WSGI response for the login page."""

    # We don't allow logging in via GET
    if query["_METHOD"] == "GET":
        # Drop user credentials
        query.pop("user_name", None)
        query.pop("user_password", None)

    # Strip unimportant stuff from query
    data = strip_query(query)

    # Has the user already provided their credentials?
    if "user_name" in query and "user_password" in query:
        # Try to log in.
        success, result = check_login(query)

        # Check for auth success.
        if success:
            return success, *result

        # Otherwise, result is the flash text
        data["flash"] = result
        data["flash_type"] = "error"
    else:
        data["flash"] = "Login required."
        data["flash_type"] = "info"

    # Invalidate the cookie and show the login page
    return render_template(
        "login",
        data=data,
        headers=[INVALIDATE_SESSION],
    )


def get_user(environ):
    """Decode the session cookie from the WSGI environment.

    Returns None if it can't be found.
    """
    if "HTTP_COOKIE" not in environ:
        return None

    cookies = http.cookies.SimpleCookie(environ["HTTP_COOKIE"])

    if "dv_session" not in cookies:
        return None

    # Returns the user if successful, or None if session
    # verification fails
    return decode_session(cookies["dv_session"].value)


def get_query(environ):
    """Parse and return the request query, if any.

    This function has primary responsibility for figuring out the client's
    request.  We accept two methods:
    * GET requests with an optional query-string in the URL
    * POST requests with application/x-www-form-urlencoded data

    If successful, returns a dict with the parsed query.  Failure, results in the
    errors 405 or 415.
    """
    method = environ["REQUEST_METHOD"]

    if method == "GET":
        query_string = environ["QUERY_STRING"]
    elif method == "POST":
        if environ["CONTENT_TYPE"] != "application/x-www-form-urlencoded":
            # Bad content-type.  415 is the standard response here
            return False, 415

        query_string = environ["wsgi.input"].read()
    else:
        # Bad method
        return False, 405

    # Parse
    try:
        encoded_query = urllib.parse.parse_qs(query_string)
    except UnicodeEncodeError:
        print(f"Unicode Encoding Error: {query_string}", file=sys.stderr)
        encoded_query = dict()

    # Decode
    query = dict()
    for key, value in encoded_query.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8")

        # We only take the first value from a list
        if isinstance(value, list):
            value = value[0]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        query[key] = value

    # Sanitise some ints (which may be missing)
    for key in ("csd", "ssd", "rev"):
        try:
            query[key] = int(query[key])
        except KeyError:
            # Missing key is fine
            pass
        except (TypeError, ValueError):
            # Drop weird values
            del query[key]

    # Append method
    query["_METHOD"] = method

    return True, query


def get_csds_for_allrevs():
    """Return a list of available CSDs for any known pipeline revision."""

    # Get a list of available renders
    csds = dict()
    for path in glob.iglob("rev??_????.html", root_dir=render_dir):
        try:
            id_ = int(path[-9:-5])
            csds[id_] = 1
        except ValueError:
            # Ignore non-numeric values
            pass

    # Return a storted list of CSDs
    return sorted(csds.keys())


def get_revs_for_csd(csd):
    """Get a list of available pipeline revisions for the specified CSD.
    The returned list is sorted ascending."""

    # Get a list of available renders
    revs = list()
    for path in glob.iglob(f"rev??_{csd:04d}.html", root_dir=render_dir):
        try:
            revs.append(int(path[3:5]))
        except ValueError:
            # Ignore non-numeric values
            pass

    return sorted(revs)


def get_opinions_for_csd(csd, revs, user):
    """Fetch the user's current opinion of CSD `csd`.

    Parameters:
    -----------
    csd : int
       The CSD to fetch opinions for; already vetted.
    revs : list of ints
        All revisions for which this CSD is valid
    user : MediaWikiUser
        The user's record

    Returns:
    -------
    opinions : dict of dicts
        A dict with keys for revision and values are
        opinion dicts.  This is typically JSON serialised
        and sent back to the client.
    """

    opinions = dict()
    for rev in revs:
        datarev = DataRevision.get(name=f"rev_{rev:02}")

        # This is not specific to the user, but we shoe-horn it in anyways
        opinion_count = (
            DataFlagOpinion.select()
            .where(
                DataFlagOpinion.revision == datarev,
                DataFlagOpinion.type == OPINION_TYPE,
                DataFlagOpinion.lsd == csd,
            )
            .count()
        )

        # Get the opinion
        try:
            opinion = DataFlagOpinion.get(
                DataFlagOpinion.revision == datarev,
                DataFlagOpinion.type == OPINION_TYPE,
                DataFlagOpinion.lsd == csd,
                DataFlagOpinion.user == user,
            )
        except pw.DoesNotExist:
            opinion = None

        if opinion:
            opinions[rev] = {
                "csd": csd,
                "decision": opinion.decision,
                "last_edit": opinion.last_edit,
                "notes": opinion.notes,
                "count": opinion_count,
            }
        else:
            opinions[rev] = {
                "csd": csd,
                "decision": "none",
                "last_edit": 0,
                "notes": "",
                "count": opinion_count,
            }

    return opinions


def no_opinion_target(source, rev, user, csdlist, previous):
    """Loop over the list of CSDs and find the "next" CSD with no
    opinion.

    Parameters
    ----------
    source: integer
        The CSD where we start looking
    rev: integer
        The revision to use
    user: MediaWikiUser
        The user's record
    csdlist: list
        The list of available CSDs, smallest first
    previous: boolean
        If True, search towards CSD 0

    Returns
    -------
    target: integer
        the chosen CSD.  If no opinionless CSDs were found, this
        is just the appropriate limit value of `csdlist`
    """

    # Bisect csdlist to find the starting location
    # Then calculate CSD search subset and limit
    if previous:
        index = bisect.bisect_left(csdlist, source)
        search_csds = list(reversed(csdlist[:index]))
        limit = csdlist[0]
    else:
        index = bisect.bisect_right(csdlist, source)
        search_csds = csdlist[index:]
        limit = csdlist[-1]

    # Short-circuit for empty search range
    if not search_csds:
        return limit

    datarev = DataRevision.get(name=f"rev_{rev:02}")

    # Subset of search_csds with opinions from this user (and this revision)
    opinedcsds = {
        opinion.lsd
        for opinion in DataFlagOpinion.select(DataFlagOpinion.lsd).where(
            DataFlagOpinion.revision == datarev,
            DataFlagOpinion.type == OPINION_TYPE,
            DataFlagOpinion.user == user,
            DataFlagOpinion.lsd << search_csds,
        )
    }

    # Now search for a non-opined CSD
    for csd in search_csds:
        if csd not in opinedcsds:
            return csd

    # Return the limit if no un-opined CSD found
    return limit


def csd_data(target, source, rev, user):
    """Fetch data for the "current" CSD.

    If a particular CSD (`target`) was specified in the request, that one will be used,
    if possible.  Otherwise, this function tries to choose a suitable CSD.

    Returns a dict of data for the chosen CSD, which is typically JSON-ified and sent
    back to the client.

    Parameters
    ----------
    target : int or str
      The requested CSD, if any, or 0 if none was requested.  May also be the special
      values "nno" or "pno" for the next/previous csd with no opinion.
    source : int
      The "source" CSD: the CSD where the user was, when requesting a new CSD
      or 0, if no source information is available.
    rev : int
      Integer pipeline revision.
    user : MediaWikiUser
      The user's record.

    Returns
    -------
    data : dict
        A dictionary of data relating to the chosen CSD.
    """
    print(f"csd_data({target}, {source}, {rev}, user)")

    # Santitise source
    try:
        source = int(source)
    except ValueError:
        source = 0

    # Sorted list of days available to the viewer.
    csdlist = get_csds_for_allrevs()

    # Santiy check.
    if not csdlist:
        raise RuntimeError(f"No CSDs found in {render_dir}!")

    # This is the dict we will return.
    selections = {"first_csd": csdlist[0], "last_csd": csdlist[-1]}

    if target == "nno" or target == "pno":
        # Set CSD for special cases.  This only works if the source is provided
        if source:
            target = no_opinion_target(
                source, rev, user, csdlist, previous=(target == "pno")
            )
        else:
            # If we don't have a starting point, then... I don't know, just go to the limit, I guess
            target = csdlist[-1] if target == "nno" else csdlist[0]
    else:
        try:
            target = int(target)
        except ValueError:
            target = 0

        if target == 0:
            # No CSD provided.

            # If SSD is valid, just use that
            # otherwise, use the latest day (ignoring revision)
            target = source if source in csdlist else csdlist[-1]
        elif target >= csdlist[-1]:  # csdlist[-1] is the largest CSD
            # Set csd to the limit, if out of bounds
            target = csdlist[-1]
        elif target <= csdlist[0]:  # csdlist[0] is the smallest CSD
            # Ditto for the lower limit
            target = csdlist[0]
        elif target not in csdlist:
            # target is in range, but if it is not available.
            if source:
                # if we know the source CSD, go to the next/previous

                # Figure out direction
                if source > target:
                    # Find previous day.  By this point we know source > target > csdlist[0]
                    # So, bisect_left will always return >= 0
                    index = bisect.bisect_left(csdlist, source)
                    target = csdlist[index]
                else:
                    # Find next day.  As with the other way, we know this will end up in-range
                    index = bisect.bisect_right(csdlist, source)
                    target = csdlist[index]
            else:
                # If we have no source info _and_ a bad CSD, just go to the end
                target = csdlist[-1]

    # Target is now set.  Get index of target
    index = csdlist.index(target)

    # Remember chosen CSD
    selections["csd"] = target

    # All revisions for this CSD
    revs = get_revs_for_csd(target)
    selections["revs"] = revs

    # Switch revisions, if necessary
    if rev not in revs:
        rev = revs[-1]
    selections["rev"] = rev

    # Determine prev and next
    if index == 0:
        selections["prev_csd"] = 0
    else:
        selections["prev_csd"] = csdlist[index - 1]

    if index == len(csdlist) - 1:
        # At the end, so zero
        selections["next_csd"] = 0
    else:
        selections["next_csd"] = csdlist[index + 1]

    # Get the user's current opinion for the selected CSD, for all available revisions
    # The opinions include the opinion count for _all_ users.
    selections["opinions"] = get_opinions_for_csd(target, revs, user)

    # Calculate the date for the selected CSD

    # Length of the sidereal day in seconds
    sidereal_day_seconds = 86400 * 0.9972697936296091
    # CSD zero point.  Seconds since the UNIX epoch when the local stellar
    # angle (LSA) was zero on the day of the CHIME Pathfinder first light
    # (2013-11-05).
    csd_zero_date = 1384489290.2213902

    csd_start = datetime.fromtimestamp(selections["csd"] * sidereal_day_seconds + csd_zero_date)
    csd_end = datetime.fromtimestamp((1 + selections["csd"]) * sidereal_day_seconds + csd_zero_date)
    selections["csd_date"] = '<a id="csd-date" class="csd-date" href="https://bao.chimenet.ca/wiki/index.php/Run_Notes_-_' + csd_end.strftime("%B_%Y#%Y-%m-%d") + '">' + csd_start.strftime("%Y-%m-%d %H:%M") + " -- " + csd_end.strftime("%Y-%m-%d %H:%M") + "</a>"

    return selections


def redirect_response(environ, headers, /, **query):
    """Generate a 302 redirect back to ourself, maybe with some query data."""

    # See PEP 3333 § URL Reconstruction
    url = environ["wsgi.url_scheme"] + "://" + environ["SERVER_NAME"]

    if environ["wsgi.url_scheme"] == "https":
        if environ["SERVER_PORT"] != "443":
            url += ":" + environ["SERVER_PORT"]
    else:
        if environ["SERVER_PORT"] != "80":
            url += ":" + environ["SERVER_PORT"]

    url += urllib.parse.quote(environ.get("SCRIPT_NAME", ""))
    url += urllib.parse.quote(environ.get("PATH_INFO", ""))

    # Include the query, if any:
    if len(query):
        url += "?"
        url += urllib.parse.urlencode(query, doseq=True)

    # 302 target
    headers.append(("Location", url))

    # Return redirect
    return 302, headers


def revision_from_query(query):
    """Returns the revision specified in the query, or
    a resonable default (which is the latest known
    revision)."""

    if "rev" not in query:
        return REVISIONS[-1]

    try:
        rev = int(query["rev"])
    except ValueError:
        return REVISIONS[-1]

    if rev not in REVISIONS:
        return REVISIONS[-1]

    return rev


def fetch_csd(user, query, extra_data=None):
    """Return data for the fetch `query` to the client.

    The data returned includes:
        * a vetted (possibly changed) CSD and revision
        * the targets of the navigation buttons
        * the user's opinions for this CSD (for all revisions)

    If `extra_data` is provided, it is merged into the generated
    data, overriding any duplicate keys.

    The data is JSON serialised and returned to the client.
    """

    # Do the actual data fetch
    data = csd_data(
        query.get("fetch", 0),
        query.get("ssd", 0),
        revision_from_query(query),
        user,
    )

    # Add the client's request ID, if provided
    if "request_id" in query:
        data["request_id"] = query["request_id"]

    # Any anything else
    if extra_data:
        data.update(extra_data)

    # Return JSON in all cases
    return 200, list(), "application/json", json.dumps(data)


def known_csd(revision, csd):
    """Is `csd` one we know about?

    A "known" csd has a file in the rendered directory.
    """

    try:
        return render_dir.joinpath(f"rev{revision:02d}_{csd:04d}.html").is_file()
    except ValueError:
        # Non-numeric csd or rev returns False
        return False


def delete_from_db(user, datarev, csd):
    """Delete the opinion (user, rev, csd), if it exists

    Also deletes categories.

    Returns a dict with the resultant message flash.
    """
    try:
        # Begin a transaction
        with db.proxy.atomic():
            # Get the current opinion ID, if any.
            # The tuple (user, rev, type, lsd) is a unique key
            opinion = DataFlagOpinion.get_or_none(
                user=user,
                revision=datarev,
                type=OPINION_TYPE,
                lsd=csd,
            )
            if opinion:
                opinion.delete_instance()

                return {"result": "good", "message": "Opinion Deleted"}
            else:
                return {"result": "good", "message": "No Change"}
    except pw.OperationalError:
        return {"result": "error", "message": "Error during DB Update!"}


def update_db(user, datarev, csd, now, decision, notes):
    """Update/add the given opinion.

    Returns a dict with the resultant message flash.
    """

    result = {"csd": csd}

    try:
        # Begin a transaction
        with db.proxy.atomic():
            # Get the current opinion ID, if any.
            # The tuple (user, datarev, type, lsd) is a unique key
            opinion = DataFlagOpinion.get_or_none(
                user=user,
                revision=datarev,
                type=OPINION_TYPE,
                lsd=csd,
            )

            if opinion:
                opinion.decision = decision
                opinion.notes = notes
                opinion.last_edit = now
                opinion.client = CLIENT
                opinion.save()

                result["result"] = "good"
                result["message"] = "Opinion Updated"
            else:
                # No update, so insert
                opinion = DataFlagOpinion.create(
                    type=OPINION_TYPE,
                    user=user,
                    decision=decision,
                    creation_time=now,
                    last_edit=now,
                    client=CLIENT,
                    revision=datarev,
                    lsd=csd,
                    notes=notes,
                )
                result["result"] = "good"
                result["message"] = "Opinion Added"
    except pw.OperationalError:
        return {"result": "error", "message": "Error during DB Update!"}

    return result


# Error return for update_opinion
def update_error(message):
    # Even though there was an error, at the HTTP level,
    # this is a scucess (=HTTP code 200)
    return (
        200,
        list(),
        "application/json",
        json.dumps({"result": "error", "message": message}),
    )


def update_opinion(user, query):
    """Upsert an opinion in the database.

    Returns a JSON payload with the result of the DB update."""

    # Unlike with a fetch, an opinion update requires a
    # properly set "rev" from the client
    if "rev" not in query or query["rev"] not in REVISIONS:
        return update_error("Bad revision")

    rev = query["rev"]
    try:
        datarev = DataRevision.get(name=f"rev_{rev:02}")
    except pw.DoesNotExist:
        return update_error("Bad revision")

    # Vet decisions
    decision = query["decision"]
    if decision not in ["none", "bad", "unsure", "good"]:
        return update_error("Bad opinion")

    try:
        notes = query["notes"]
        if not notes:
            notes = None
    except KeyError:
        notes = None

    # Convert CSD
    try:
        csd = int(query["csd"])
        # Do we know about this CSD?
        if not known_csd(rev, csd):
            csd = 0
    except (KeyError, TypeError):
        csd = 0

    # Validate required fields
    if csd <= 0:
        return update_error("Bad CSD")

    # This will get JSON-ified
    result = dict(csd=csd, decision=decision)

    if decision == "none":
        # A request to delete an opinion (if it exists).
        result.update(delete_from_db(user, rev, csd))
    else:
        # A request to update or insert an opinion.
        now = time.time()

        if notes:
            result["notes"] = notes
        else:
            result["notes"] = ""

        result.update(update_db(user, rev, csd, now, decision, notes))

    # On error, just return the error
    if result["result"] == "error":
        return update_error(result["message"])

    # otherwise, re-fetch the opinion from the DB and return it to the client
    result.update(csd_data(csd, 0, rev, user))

    # Add the client's request ID, if provided
    if "request_id" in query:
        result["request_id"] = query["request_id"]

    # Return JSON in all cases
    return 200, list(), "application/json", json.dumps(result)


def get_response(environ):
    """Generate the response to the request.

    Returns
    -------
    status: int
        an integer HTTP status code.
    headers: list
        additional header tuples
    content_type: str
        the content type of the returned payload.  Ignored if status != 200
    payload: list of encoded strings
        the response payload.  Ignored if status != 200.
    """

    headers = list()

    # Vet the request
    result, query = get_query(environ)
    if not result:
        # On error, `query` holds the HTTP status
        return query, [("Accept", "application/x-www-form-urlencoded")]

    if "logout" in query:
        # Logout
        headers.append(INVALIDATE_SESSION)

        # Redirect back to self, maybe with a pointer to where we were
        return redirect_response(environ, headers, **strip_query(query))

    # Connect to DB
    db.connect(read_write=True)

    # Check for login
    user = get_user(environ)
    if not user:
        # If this is a fetch, render or decision push, just return 403 and invalidate
        # the session
        if "fetch" in query or "render" in query or "decision" in query:
            return 403, [INVALIDATE_SESSION]

        # Returns (True, user) on successful login, or the 4-tuple response
        # on failure
        result = do_login(query)

        # Check for login success
        if result[0] is not True:
            # As pointed out above, in this case `result` is a full 4-tuple response.
            return result

        # Otherwise, result[1] is the user
        user = result[1]

        # result[2] is the set-cookie header
        headers.append(result[2])

        # Perform a redirect to cleanse the request of login details.
        # This also allows us to verify that the client has correctly set
        # the session cookie
        return redirect_response(environ, headers, **strip_query(query))

    # If we get here, we're logged in, and both user and query are valid.
    set_globals()

    # What kind of query is this?
    if "fortnight" in query:
        # Render the 2-week view
        return render_template(
            "fortnight",
            data={
                "csd": query.get("csd", 0),
                "rev": query.get("rev", 0),
                "ui_class": "ui_2week",
            },
            headers=headers,
        )
    if "render" in query:
        # A request for a rendered notebook
        if "/" in query["render"]:
            # Path elements aren't allowed in the render request
            return 403, headers

        # Returns an HTML file from the render_dir
        try:
            with open(render_dir.joinpath(f"{query['render']}.html"), "rb") as f:
                return 200, headers, "text/html; charset=utf-8", [f.read()]
        except (FileNotFoundError, PermissionError):
            return 404, headers
    if "fetch" in query:
        # A CSD data fetch
        return fetch_csd(user, query)
    if "decision" in query:
        # An opinion update
        return update_opinion(user, query)

    # Otherwise, this is just a simple GET for a viewer page.

    # Template data
    data = {
        "revisions": REVISIONS,
        "logout": random.choice(
            ["yes", "ok", "okay", "now", "please", "do_it", "logout"]
        ),
    }

    # Choose the first CSD to display on page load
    data.update(
        csd_data(
            query.get("csd", 0),
            query.get("ssd", 0),
            revision_from_query(query),
            user,
        )
    )

    # Render the view
    return render_template("view", data=data, headers=headers)


def application(environ, start_response):
    """Entry-point for the WSGI application."""

    # HTTP status codes
    http_status = {
        200: "200 OK",
        302: "302 Found",
        403: "403 Forbidden",
        404: "404 Not Found",
        405: "405 Bad Method",
        415: "415 Unsupported Media Type",
        500: "500 Internal Server Error",
        501: "501 Not Implemented",
    }

    # Generate response
    response = get_response(environ)

    # Unpack response and handle errors
    if response[0] != 200:
        status, headers = response
        content_type = "text/plain; charset=us-ascii"
        payload = [http_status[status]]
    else:
        status, headers, content_type, payload = response

    # Add content type to headers
    if headers is None:
        headers = [("Content-Type", content_type)]
    else:
        headers.append(("Content-Type", content_type))

    # Encode the payload, if necessary
    encoded_payload = list()
    content_length = 0
    for item in payload:
        if isinstance(item, bytes):
            pass
        elif isinstance(item, str):
            item = item.encode("utf-8")
        else:
            raise TypeError(f"Don't know what to do with {item} ({type(item)})")
        encoded_payload.append(item)
        content_length += len(item)

    headers.append(("Content-Length", str(content_length)))

    # Return WSGI response
    start_response(http_status[status], headers)
    return encoded_payload
