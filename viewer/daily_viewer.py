#!/usr/bin/env python3

import sys
import glob
import json
import time
import base64
import jinja2
import random
import hashlib
import pathlib
import http.cookies
import peewee as pw
import urllib.parse
import chimedb.core as db
from chimedb.core.mediawiki import MediaWikiUser
from chimedb.dataflag import (
    DataFlagClient,
    DataFlagOpinion,
    DataFlagOpinionType,
    DataRevision,
)

__version__ = "v0.1"

# Should eventually be a settable parameter
_REVISION = 7

# Header to invalidate the session cookie
INVALIDATE_SESSION = ("Set-Cookie", "dv_session=; SameSite=Strict; Secure; Max-Age=0;")

# The DataFlagOpinionType and DataFlagClient we're using.  These are set on first use
OPINION_TYPE = None
CLIENT = None

# Directories
template_dir = pathlib.Path(__file__).with_name("templates")
web_dir = pathlib.Path(__file__).with_name("web")
render_dir = pathlib.Path(__file__).with_name("rendered")

# Our URL path
script_name = ""

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
            client_name="daily_viewer", client_version=__version__
        )


def render_template(name, data, headers=None):
    """Render a Jinja2 template `name` using `data`."""

    # These are always part of the data
    data["script_name"] = script_name
    data["base_path"] = str(pathlib.Path(script_name).parent)

    template = jinja_env.get_template(name + ".html.j2")
    return 200, headers, "text/html", template.render(**data)


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
        if int(parts[1]) + 86400 <= time.time():
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


def do_login(query):
    """Ask for, and perform login.

    On successful login, returns a tuple (True, user, cookie).
    On failure returns a response providing the login page."""

    # We don't allow logging in via GET
    if query["_METHOD"] == "GET":
        # Drop user credentials
        query.pop("user_name", None)
        query.pop("user_password", None)

    # Remember csd, if given
    csd = query.get("csd", 0)

    # Are we in two-week mode?
    fortnight = ("fortnight" in query)

    if "user_name" in query and "user_password" in query:
        # Try to log in.  `success`
        success, result = check_login(query)

        # Check for auth success.
        if success:
            return success, *result

        # Otherwise, result is the flash text
        flash = result
        flash_type = "error"
    else:
        flash = "Login required."
        flash_type = "info"

    # Data
    data={"csd": csd, "flash": flash, "flash_type": flash_type}
    if fortnight:
        data["fortnight"]="yes"

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

    # Sanitise
    try:
        query["csd"] = int(query["csd"])
    except KeyError:
        # No CSD is fine
        pass
    except (TypeError, ValueError):
        # Drop weird CSD values
        del query["csd"]

    # Append method
    query["_METHOD"] = method

    return True, query


def render_opinion_json(csds):
    """Convert the list returned by get_csds, into JSON."""

    pre_json = list()

    for day in reversed(sorted(csds.keys())):
        if csds[day] is None:
            pre_json.append(["none", "", 0])
        else:
            notes = "" if csds[day].notes is None else csds[day].notes
            pre_json.append([csds[day].decision, notes, csds[day].last_edit])

    return json.dumps(pre_json)


def get_csds(user, revision):
    """Get the list of available CSDs and the user's opinions, if any"""

    # First get a list of available renders
    csds = dict()
    for path in glob.iglob(f"rev{revision:02d}_????.html", root_dir=render_dir):
        try:
            # None as a value here indicates no opinion for this user for this day
            csds[int(path[-9:-5])] = None
        except ValueError:
            # Ignore non-numeric values
            pass
    print(
        f"daily_viewer: {len(csds)} pages in render_dir={render_dir}", file=sys.stderr
    )

    set_globals()
    rev = DataRevision.get(name=f"rev_{_REVISION:02}")

    # Now fetch opinions, if any, for this user:
    query = DataFlagOpinion.select().where(
        DataFlagOpinion.type == OPINION_TYPE,
        DataFlagOpinion.revision == revision,
        DataFlagOpinion.user == user,
        DataFlagOpinion.lsd << list(csds.keys()),
    )

    for opinion in query.execute():
        csds[opinion.lsd] = opinion

    return csds


def csd_vars(query, csds):
    """Determine which CSD to display on first load of the viewer.

    If one was specified in the request, that one will be used, if possible.

    After CSD selection, determine the values of the six selector buttons."""

    # A list of CSDs in descending order
    csdlist = list(reversed(sorted(csds.keys())))

    # This is the dict we will return
    selections = {"csd_list": json.dumps(csdlist)}

    # If there is no CSD specified in the request, then
    # find the newest CSD with no opinion.  If that fails,
    # select the newest CSD
    if "csd" not in query or query["csd"] == 0:
        for csd in csdlist:
            if csds[csd] is None:
                break
        else:
            # Just use the last csd if we couldn't find one
            csd = csdlist[0]
    else:
        # User specified a CSD
        csd = query["csd"]

        # Set csd to the limits, if out of bounds
        if csd > csdlist[0]:  # csdlist[0] is the largest CSD
            csd = csdlist[0]
        elif csd < csdlist[-1]:  # csdlist[-1] is the smallest CSD
            csd = csdlist[-1]
        else:
            # csd is in range, but if it is not available,
            # go to the next day available
            if csd not in csdlist:
                for i in range(len(csdlist) - 1):
                    if csd > csdlist[i + 1]:  # True when csd[i] > csd > csd[i + 1]
                        csd = csdlist[i]
                        break

    # Remember chosen CSD
    selections["csd"] = csd

    # Index of CSD
    index = csdlist.index(csd)

    # These are easy
    selections["first_csd"] = csdlist[-1]
    selections["last_csd"] = csdlist[0]

    # Determine Prev and PNO
    if csd == csdlist[-1]:
        # At the first CSD, so both are zero
        selections["prev_csd"] = 0
        selections["pno_csd"] = 0
    else:
        selections["prev_csd"] = csdlist[index + 1]

        # Find a CSD with no opinion in the trailing part of the list:
        for pno in csdlist[index + 1 :]:
            if csds[pno] is None:
                selections["pno_csd"] = pno
                break
        else:
            # None found
            selections["pno_csd"] = 0

    # Determine Next and NNO
    if index == 0:
        # At the end, so both are zero
        selections["next_csd"] = 0
        selections["nno_csd"] = 0
    else:
        selections["next_csd"] = csdlist[index - 1]

        # Find a CSD with no opinion in the leading part of the list:
        for nno in csdlist[index - 1 :: -1]:
            if csds[nno] is None:
                selections["nno_csd"] = nno
                break
        else:
            # None found
            selections["nno_csd"] = 0

    # The current opinion for the selected CSD
    if csds[csd] is None:
        selections["csd_decision"] = "none"
        selections["csd_notes"] = ""
    else:
        selections["csd_decision"] = csds[csd].decision
        selections["csd_notes"] = "" if csds[csd].notes is None else csds[csd].notes

    return selections


def redirect_response(environ, headers, /, **query):
    """Generate a 302 redirect back to ourself, maybe with some query data."""

    # See PEP 3333 § URL Reconstruction
    url = environ["wsgi.url_scheme"] + "://"

    if environ.get("HTTP_HOST"):
        url += environ["HTTP_HOST"]
    else:
        url += environ["SERVER_NAME"]

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


def update_opinion(user, query):
    """Upsert an opinion in the database.

    Returns a JSON payload with the result of the DB update."""

    set_globals()
    rev = DataRevision.get(name=f"rev_{_REVISION:02}")

    # Valid decisions
    all_decisions = ["none", "bad", "unsure", "good"]

    decision = query["decision"]

    try:
        notes = query["notes"]
        if not notes:
            notes = None
    except KeyError:
        notes = None

    # Convert CSD
    try:
        csd = int(query["csd"])
    except (KeyError, TypeError):
        csd = 0

    # This will get JSON-ified
    result = dict(csd=csd, decision=decision)

    # Validate required fields
    if csd <= 0:
        result["result"] = "error"
        result["message"] = "Bad CSD"
    elif decision not in all_decisions:
        result["result"] = "error"
        result["message"] = "Bad Opinion"
    elif decision == "none":
        # This is a request to delete an opinion.
        #
        # The tuple (user, rev, type, lsd) is a unique key, so there can only
        # ever be at most one record deleted, but we put a limit on, just in case
        query = (
            DataFlagOpinion.delete()
            .where(
                DataFlagOpinion.user == user,
                DataFlagOpinion.revision == rev,
                DataFlagOpinion.type == OPINION_TYPE,
                DataFlagOpinion.lsd == csd,
            )
            .limit(1)
        )
        print(query.sql(), file=sys.stderr)
        query.execute()
        result["result"] = "good"
        result["message"] = "Opinion Deleted"
    else:
        # A request to update or insert an opinion.
        now = time.time()

        if notes:
            result["notes"] = notes
        else:
            result["notes"] = ""

        query = (
            DataFlagOpinion.update(
                decision=decision, notes=notes, last_edit=now, client=CLIENT
            )
            .where(
                DataFlagOpinion.user == user,
                DataFlagOpinion.revision == rev,
                DataFlagOpinion.type == OPINION_TYPE,
                DataFlagOpinion.lsd == csd,
            )
            .limit(1)
        )
        print(query.sql(), file=sys.stderr)
        count = query.execute()
        if count == 0:
            # No update, so insert
            query = DataFlagOpinion.insert(
                type=OPINION_TYPE,
                user=user,
                decision=decision,
                creation_time=now,
                last_edit=now,
                client=CLIENT,
                revision=rev,
                lsd=csd,
                notes=notes,
            )
            print(query.sql(), file=sys.stderr)
            query.execute()
            result["result"] = "good"
            result["message"] = "Opinion Added"
        else:
            result["result"] = "good"
            result["message"] = "Opinion Updated"

    # Done, return JSON in all cases
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
    data = dict()

    # Vet the request
    result, query = get_query(environ)
    if not result:
        # On error, `query` holds the HTTP status
        return query, [("Accept", "application/x-www-form-urlencoded")]

    if "logout" in query:
        # Logout
        headers.append(INVALIDATE_SESSION)

        # New query for the redirect
        new_query = dict()
        if "csd" in query and query["csd"]:
            new_query["csd"] = query["csd"]
        if "fortnight" in query:
            new_query["fortnight"] = "yes"

        # Redirect back to self, maybe with a CSD
        return redirect_response(environ, headers, **new_query)

    # Script name
    global script_name
    script_name = environ.get("SCRIPT_NAME")

    # Connect to DB
    db.connect(read_write=True)

    # Check for login
    user = get_user(environ)
    if not user:
        # If this is a fetch or opinion push, just return 403 and invalidate
        # the session
        if "fetch" in query or "decision" in query:
            return 403, [INVALIDATE_SESSION]

        # Returns (True, user) on successful login, or the 4-tuple response
        # on failure
        result = do_login(query)

        # Check for login success
        if result[0] is not True:
            return result

        # Otherwise, result[1] is the user
        user = result[1]

        # result[2] is the set-cookie header
        headers.append(result[2])

        # New query for the redirect
        new_query = dict()
        if "csd" in query and query["csd"]:
            new_query["csd"] = query["csd"]
        if "fortnight" in query:
            new_query["fortnight"] = "yes"

        # Perform a redirect to cleanse the request of login details.
        # This also allows us to verify that the client has correctly set
        # the session cookie
        return redirect_response(environ, headers, **new_query)

    # If we get here, we're logged in, and both user and query are valid.
    if "fortnight" in query:
        # Render the 2-week view
        return render_template(
            "fortnight",
            data={"csd": query.get("csd", 0), "ui_class": "ui_2week"},
            headers=headers,
        )
    if "fetch" in query:
        # Returns an HTML file from the render_dir
        try:
            with open(render_dir.joinpath(f"{query['fetch']}.html"), "rb") as f:
                return 200, headers, "text/html", [f.read()]
        except (FileNotFoundError, PermissionError):
            return 404, headers
    elif "decision" in query:
        # An opinion update
        return update_opinion(user, query)

    # Available days with opinions (if any)
    csds = get_csds(user, _REVISION)
    data["opinions"] = render_opinion_json(csds)

    # Choose the first CSD to display on page load
    data.update(csd_vars(query, csds))

    # Logout
    data["logout"] = random.choice(
        ["yes", "ok", "okay", "now", "please", "do_it", "logout"]
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
        content_type = "text/plain"
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
    for item in payload:
        if isinstance(item, bytes):
            encoded_payload.append(item)
        elif isinstance(item, str):
            encoded_payload.append(item.encode("utf-8"))
        else:
            raise TypeError(f"Don't know what to do with {item} ({type(item)})")

    # Return WSGI response
    start_response(http_status[status], headers)
    return encoded_payload
