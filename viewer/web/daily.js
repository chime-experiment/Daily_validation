function daily_logout(how) { window.location.href = '/daily/view?logout=' + how + '&csd=' + csd }

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

function update_ui(opinion) {
  console.log("update_ui: " + csd)

  // Update the document
  document.title = "CHIME daily viewer - CSD " + csd
  window.history.pushState(document.getElementById("root").innerHTML, "", "/daily/view?csd=" + csd)

  // Clear flash
  clear_flash()

  // Load new render
  document.getElementById("frame").src = "/daily/view?fetch=rev07_" + csd

  // Enable/disable buttons
  set_disable("button_first", csd == first_csd)
  set_disable("button_pno", pno_csd == 0)
  set_disable("button_prev", prev_csd == 0)
  set_disable("button_next", next_csd == 0)
  set_disable("button_nno", next_csd == 0)
  set_disable("button_last", csd == last_csd)

  // Set selector input
  document.getElementById("CSD").value = csd

  // Set opinion title
  document.getElementById("opinion-h2").innerHTML = "Opinion for CSD #" + csd

  // Set opinion decision
  var decision = opinion[0]

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
    document.getElementById("opinion-notes").value = opinion[1]
    set_disable("opinion-notes", false)
  }
}

function vet_csd() {
  var new_csd = +(document.getElementById("CSD").value)

  // Ignore non-numbers
  if (new_csd != new_csd) {
    new_csd = csd
  }

  // Has anything changed?
  if (new_csd != csd) {
    // Check for a valid CSD
    if (csd_list.indexOf(new_csd) === -1) {
      // Bad input.  Try to find the right one
      if (new_csd > last_csd) {
        // Past end
        new_csd = last_csd
      } else if (new_csd < first_csd) {
        // Before start
        new_csd = first_csd
      } else if (new_csd > csd) {
        // User is going up
        for (; new_csd <= last_csd; new_csd++) {
          if (csd_list.indexOf(new_csd) != -1) {
            break
          }
        }
      } else {
        // User is going down
        for (; new_csd >= first_csd; new_csd--) {
          if (csd_list.indexOf(new_csd) != -1) {
            break
          }
        }
      }
    }
  }

  // Now set
  set_csd(new_csd)
}

function set_csd(csd_in) {
  // coerce to number
  var new_csd = +csd_in

  // Reject unknown CSDs
  var index = csd_list.indexOf(new_csd)

  if (index === -1) {
    index = csd_list.indexOf(new_csd)
  } else {
    csd = new_csd
  }

  // Update next and nno
  if (index === 0) {
    next_csd = 0
    nno_csd = 0
  } else {
    next_csd = csd_list[index - 1]
    // Search backwards through array
    for (var nno = index - 1; nno >= 0; --nno) {
      if (opinions[nno][0] == "none") {
        nno_csd = csd_list[nno]
        break
      }
    }
    if (nno < 0) {
      nno_csd = 0
    }
  }

  // Update prev and pno
  if (index == csd_list.length - 1) {
    prev_csd = 0
    pno_csd = 0
  } else {
    prev_csd = csd_list[index + 1]
    // Search forwards through array
    for (var pno = index + 1; pno < csd_list.length; ++pno) {
      if (opinions[pno][0] == "none") {
        pno_csd = csd_list[pno]
        break
      }
    }
    if (pno == csd_list.length) {
      pno_csd = 0
    }
  }

  return update_ui(opinions[index])
}

function submit_opinion() {
  // Get current opinion

  var index = csd_list.indexOf(csd)
  if (index === -1) {
    // Do noting if we're not on a valid CSD
    return
  }

  // The current opinion
  var decision = opinions[index][0]
  var notes = opinions[index][1]

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

  // Assemble the POST data
  var post_data = "csd=" + csd + "&decision=" + new_decision + "&notes=" + encodeURIComponent(new_notes).replace(/%20/g, "+")
  console.log("data: " + post_data)

  // Create the XHR
  const xhr = new XMLHttpRequest()

  xhr.addEventListener("load", (event) => {
    if (event.target.status != 200) {
      set_flash("warning", "Server Response " + event.target.status + ".  Check logs")
    } else {
      var result = JSON.parse(event.target.responseText)

      console.log("result: " + result)

      // Update opinion
      if (result.result != "error") {
        var index = csd_list.indexOf(result.csd)
        console.log("index: " + index)
        if (index != -1) {
          opinions[index][0] = result.decision
          opinions[index][1] = result.notes
          console.log("opinion: " + opinions[index])
        }
      }

      // Report result
      set_flash(result.result, result.message)
    }
  })
  xhr.addEventListener("error", (event) => {
    set_flash("error", "Submission error! Check logs.")
  })

  // POST data
  xhr.open("POST", "/daily/view")
  xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  xhr.send(post_data)
}
