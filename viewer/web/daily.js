// This is used to track asynchrous fetch requests.
var request_id = 0;

function daily_logout(how) { location.assign(window.location.pathname + '?logout=' + how + '&csd=' + csd + '&rev=' + rev) }

function fortnight(how) { location.assign(window.location.pathname + '?fortnight=' + how + '&csd=' + csd + '&rev=' + rev) }

function show_day() { location.assign(window.location.pathname + '?csd=' + csd + '&rev=' + rev) }

// returns a string of the form "rev07"
function revname(rev) {
  srev = "0" + rev
  return "rev" + srev.slice(-2)
}

function set_disable(id, disable) {
  elem = document.getElementById(id)
  if (disable) {
    elem.setAttribute("disabled", "disabled")
  } else {
    elem.removeAttribute("disabled")
  }
}

function set_flash(type, value) {
  flash = document.getElementById("flash")

  flash.className = "flash flash-" + type
  flash.innerHTML = value
}

function clear_flash() {
  set_flash("empty", "")
}

// Receives data from a fetch or decision XHR and updates the globals
// Then calls draw_ui to update the document
function update_data(result) {
  console.log("update_ui: " + JSON.stringify(result))

  // Verify the request id
  if (result.request_id != request_id) {
    console.log("Ignoring response with invalid request ID: " + result.request_id + " != " + request_id)
    return
  }

  // Update the opinion data
  if ("opinions" in result) {
    opinions = result.opinions
  }

  // Are we on a new "page"?
  var csd_changed = ("csd" in result && result.csd != csd)
  var rev_changed = ("rev" in result && result.rev != rev)

  // Update the document
  if (csd_changed) { csd = result.csd }
  if (rev_changed) { rev = result.rev }

  // Update the nav button targets
  if ("first_csd" in result) { first_csd = result.first_csd }
  if ("prev_csd" in result)  { prev_csd = result.prev_csd }

  if ("last_csd" in result) { first_csd = result.first_csd }
  if ("next_csd" in result)  { next_csd = result.next_csd }

  draw_ui(csd_changed || rev_changed)
}

// Update the UI based on the contents of the globals
function draw_ui(view_changed) {
  console.log("draw_ui: " + view_changed)
  // Clear flash
  clear_flash()

  // Update title, and history, if necessary
  if (view_changed) {
    document.title = "CHIME daily viewer - CSD " + csd + " revision " + rev
    window.history.pushState(document.getElementById("root").innerHTML, "", window.location.pathname + "?csd=" + csd + "&rev=" + rev)

    // Load new render
    document.getElementById("frame").src = window.location.pathname + "?render=" + revname(rev) + "_" + csd
  }

  // Update revision buttons
  revisions.forEach((pickrev) => {
    button = document.getElementById("pickrev-" + pickrev)
    if (pickrev in opinions) {
      decision = opinions[pickrev].decision
      button.removeAttribute("disabled")
    } else {
      decision = "missing"
      button.setAttribute("disabled", "disabled")
    }
    if (pickrev == rev) {
      selected = " pickrev-selected"
    } else {
      selected = ""
    }
    button.className = "pickrev pickrev-" + decision + selected
  });


  // Update the vote count
  var opinion = opinions[rev]
  document.getElementById("opinion-count").innerHTML = opinion.count + " existing votes"

  // Enable/disable buttons
  set_disable("button_first", csd == first_csd)
  set_disable("button_pno", csd == first_csd)
  set_disable("button_prev", prev_csd == 0)
  set_disable("button_next", next_csd == 0)
  set_disable("button_nno", csd == last_csd)
  set_disable("button_last", csd == last_csd)

  // Set selector input
  document.getElementById("CSD").value = csd

  // Set opinion title
  document.getElementById("opinion-h2").innerHTML = "Opinion for CSD #" + csd + ", rev " + rev

  // Set opinion decision
  var decision = opinion.decision

  // Sanitise
  if (decision != "good" && decision != "bad" && decision != "unsure") {
    decision = "none"
  }

  if (decision == "none") {
    document.getElementById("opinion-none").checked = true
    document.getElementById("opinion-notes").value = ""
    set_disable("opinion-notes", true)
  } else {
    document.getElementById("opinion-" + decision).checked = true
    document.getElementById("opinion-notes").value = opinion.notes
    set_disable("opinion-notes", false)
  }
}

// Called whenever a user clicks a revision button
function set_rev(rev_in) {
  rev = rev_in
  prefrev = rev_in
  draw_ui(true)
}

// Called whenever a new CSD is requested
function set_csd(csd_in) {
  var new_csd = csd_in;
  var render_early = 1;

  // Handle special cases
  if (csd_in == "pno" || csd_in == "nno") {
    render_early = 0;
  } else {
    // coerce to number
    new_csd = +csd_in

    // Ignore non-numbers
    if (new_csd != new_csd) {
      new_csd = csd
    }

    // Has anything changed?
    if (new_csd == csd) {
      // If not, do nothing
      return
    }
  }

  // Create a request for data from the server

  // Increment request_id
  request_id = request_id + 1

  // Assemble the POST data.   We always use the "preferred revision" here
  var post_data = "fetch=" + new_csd + "&ssd=" + csd + "&request_id=" + request_id + "&rev=" + prefrev

  // Create the XHR
  const xhr = new XMLHttpRequest()

  xhr.addEventListener("load", (event) => {
    if (event.target.status != 200) {
      set_flash("warning", "Server Response " + event.target.status + ".  Check logs")
    } else {
      var result;
      try {
        result = JSON.parse(event.target.responseText)
      } catch(error) {
        set_flash("error", "Bad response from server! Check logs.")
        throw(error)
      }

      // Update globals and redraw UI
      update_data(result)
    }
  })
  xhr.addEventListener("error", (event) => {
    set_flash("error", "Request error! Check logs.")
  })

  // POST data
  xhr.open("POST", window.location.pathname)
  xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  xhr.send(post_data)

  if (render_early) {
    // Assume the new CSD is correct, and pre-emptively load the new render
    document.getElementById("frame").src = window.location.pathname + "?render=" + revname(prefrev) + "_" + csd
  }

  // Nothing else to do until the XHR returns
}

function submit_opinion() {
  // Get current opinion
  var decision = opinions[rev].decision
  var notes = opinions[rev].notes

  var new_decision = decision
  if (document.getElementById("opinion-none").checked) {
    new_decision = "none"
  } else if (document.getElementById("opinion-unsure").checked) {
    new_decision = "unsure"
  } else if (document.getElementById("opinion-good").checked) {
    new_decision = "good"
  } else if (document.getElementById("opinion-bad").checked) {
    new_decision = "bad"
  }

  var new_notes = document.getElementById("opinion-notes").value

  if (decision == new_decision && notes == new_notes) {
    // Nothing changed
    set_flash("info", "No change.")
    return
  }

  // Increment request_id
  request_id = request_id + 1

  // Assemble the POST data
  var post_data = "csd=" + csd + "&rev=" + rev + "&decision=" + new_decision + "&request_id=" + request_id
    + "&notes=" + encodeURIComponent(new_notes).replace(/%20/g, "+")

  // Create the XHR
  const xhr = new XMLHttpRequest()

  xhr.addEventListener("load", (event) => {
    if (event.target.status != 200) {
      set_flash("warning", "Server Response " + event.target.status + ".  Check logs")
    } else {
      var result = JSON.parse(event.target.responseText)

      // Update opinion
      if (result.result != "error") {
        update_data(result)
      } else {
        // Report error
        set_flash(result.result, result.message)
      }
    }
  })
  xhr.addEventListener("error", (event) => {
    set_flash("error", "Submission error! Check logs.")
  })

  // POST data
  xhr.open("POST", window.location.pathname)
  xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  xhr.send(post_data)
}
