(function () {
  "use strict";

  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const ranks = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const state = {
    frames: [],
    selectedIndex: -1,
    selectedGameKey: null,
    orientation: "white",
    playing: false,
    speedMs: 700,
    timer: null
  };

  const el = {
    board: document.getElementById("board"),
    filesTop: document.getElementById("filesTop"),
    filesBottom: document.getElementById("filesBottom"),
    ranksLeft: document.getElementById("ranksLeft"),
    ranksRight: document.getElementById("ranksRight"),
    runLabel: document.getElementById("runLabel"),
    positionTitle: document.getElementById("positionTitle"),
    sideToMove: document.getElementById("sideToMove"),
    plyLabel: document.getElementById("plyLabel"),
    outcomeLabel: document.getElementById("outcomeLabel"),
    gameList: document.getElementById("gameList"),
    moveList: document.getElementById("moveList"),
    searchList: document.getElementById("searchList"),
    eventList: document.getElementById("eventList"),
    activityList: document.getElementById("activityList"),
    rootValue: document.getElementById("rootValue"),
    legalMoves: document.getElementById("legalMoves"),
    stepBack: document.getElementById("stepBack"),
    playPause: document.getElementById("playPause"),
    stepForward: document.getElementById("stepForward"),
    jumpLive: document.getElementById("jumpLive"),
    flipBoard: document.getElementById("flipBoard"),
    speedRange: document.getElementById("speedRange")
  };

  function gameKey(frame) {
    if (frame.game_index === null || frame.game_index === undefined) {
      return null;
    }
    return [
      frame.phase || "selfplay",
      frame.cycle === null || frame.cycle === undefined ? "0" : String(frame.cycle),
      String(frame.game_index)
    ].join(":");
  }

  function hasBoard(frame) {
    return Boolean(frame && (frame.after || frame.before));
  }

  function currentFrame() {
    if (state.selectedIndex < 0 || state.selectedIndex >= state.frames.length) {
      return null;
    }
    return state.frames[state.selectedIndex];
  }

  function currentGameFrames() {
    return state.frames.filter(function (frame) {
      return gameKey(frame) === state.selectedGameKey && hasBoard(frame);
    });
  }

  function addFrame(frame) {
    state.frames.push(frame);
    const key = gameKey(frame);
    if (key && !state.selectedGameKey) {
      state.selectedGameKey = key;
    }
    if (key && hasBoard(frame) && (state.playing || state.selectedIndex === state.frames.length - 2 || state.selectedIndex === -1)) {
      state.selectedGameKey = key;
      state.selectedIndex = state.frames.length - 1;
    }
    render();
  }

  function setSelectedFrameByGameOffset(offset) {
    const frames = currentGameFrames();
    if (!frames.length) {
      return;
    }
    const bounded = Math.max(0, Math.min(offset, frames.length - 1));
    state.selectedIndex = state.frames.indexOf(frames[bounded]);
    render();
  }

  function selectedOffsetInGame() {
    const frames = currentGameFrames();
    const frame = currentFrame();
    if (!frame) {
      return -1;
    }
    return frames.indexOf(frame);
  }

  function render() {
    renderCoordinates();
    renderBoard();
    renderStatus();
    renderGames();
    renderMoves();
    renderSearch();
    renderEvents();
    renderActivity();
  }

  function orientedFiles() {
    return state.orientation === "white" ? files : files.slice().reverse();
  }

  function orientedRanks() {
    return state.orientation === "white" ? ranks.slice().reverse() : ranks;
  }

  function renderCoordinates() {
    const visibleFiles = orientedFiles();
    const visibleRanks = orientedRanks();
    el.filesTop.innerHTML = "<span></span>" + visibleFiles.map(labelSpan).join("") + "<span></span>";
    el.filesBottom.innerHTML = "<span></span>" + visibleFiles.map(labelSpan).join("") + "<span></span>";
    el.ranksLeft.innerHTML = visibleRanks.map(labelSpan).join("");
    el.ranksRight.innerHTML = visibleRanks.map(labelSpan).join("");
  }

  function labelSpan(value) {
    return "<span>" + escapeHtml(value) + "</span>";
  }

  function renderBoard() {
    const frame = currentFrame();
    const snapshot = frame ? frame.after || frame.before : null;
    const piecesBySquare = new Map();
    if (snapshot && Array.isArray(snapshot.pieces)) {
      snapshot.pieces.forEach(function (piece) {
        if (piece.square) {
          piecesBySquare.set(piece.square, piece);
        }
      });
    }
    const lastSquares = new Set();
    if (frame && frame.move) {
      lastSquares.add(frame.move.from);
      lastSquares.add(frame.move.to);
    }
    const deltaSquares = new Set();
    if (frame && frame.delta && Array.isArray(frame.delta.changed_squares)) {
      frame.delta.changed_squares.forEach(function (square) {
        deltaSquares.add(square);
      });
    }

    const html = [];
    orientedRanks().forEach(function (rank) {
      orientedFiles().forEach(function (file) {
        const square = file + rank;
        const fileIndex = files.indexOf(file);
        const rankIndex = ranks.indexOf(rank);
        const shade = (fileIndex + rankIndex) % 2 === 0 ? "dark" : "light";
        const classes = ["square", shade];
        if (lastSquares.has(square)) {
          classes.push("last");
        }
        if (deltaSquares.has(square)) {
          classes.push("delta");
        }
        html.push("<div class=\"" + classes.join(" ") + "\" role=\"gridcell\" aria-label=\"" + square + "\">");
        html.push("<span class=\"coord\">" + square + "</span>");
        const piece = piecesBySquare.get(square);
        if (piece) {
          html.push(renderPiece(piece));
        }
        html.push("</div>");
      });
    });
    el.board.innerHTML = html.join("");
  }

  function renderPiece(piece) {
    const title = [
      piece.color,
      piece.class_name,
      piece.id,
      piece.square
    ].filter(Boolean).join(" ");
    const badge = piece.display_letter
      ? "<span class=\"badge\">" + escapeHtml(piece.display_letter) + "</span>"
      : "";
    return "<span class=\"piece " + escapeHtml(piece.color) + "\" title=\"" + escapeHtml(title) + "\">" +
      escapeHtml(piece.symbol || "?") +
      badge +
      "</span>";
  }

  function renderStatus() {
    const frame = currentFrame();
    const snapshot = frame ? frame.after || frame.before : null;
    const phase = frame ? frame.phase || "viewer" : "viewer";
    const cycle = frame && frame.cycle ? " cycle " + frame.cycle : "";
    const game = frame && frame.game_index !== null && frame.game_index !== undefined
      ? " game " + (frame.game_index + 1)
      : "";
    el.runLabel.textContent = phase + cycle + game;
    el.positionTitle.textContent = frame ? titleForFrame(frame) : "PixieChess Viewer";
    el.sideToMove.textContent = "side: " + (snapshot ? snapshot.side_to_move : "-");
    el.plyLabel.textContent = "ply: " + (frame && frame.ply !== null && frame.ply !== undefined ? frame.ply : "-");
    el.outcomeLabel.textContent = "outcome: " + (frame && frame.outcome ? frame.outcome : "-");
  }

  function titleForFrame(frame) {
    if (frame.event === "game_completed") {
      return "Game complete";
    }
    if (frame.move) {
      return frame.move.label + " by " + frame.move.piece_id;
    }
    if (frame.event === "game_started") {
      return "Game started";
    }
    return frame.event.replace(/_/g, " ");
  }

  function renderGames() {
    const latestByGame = new Map();
    state.frames.forEach(function (frame) {
      const key = gameKey(frame);
      if (key && hasBoard(frame)) {
        latestByGame.set(key, frame);
      }
    });
    const rows = Array.from(latestByGame.entries()).slice(-30).reverse();
    if (!rows.length) {
      el.gameList.innerHTML = "<p class=\"empty\">Waiting for a game.</p>";
      return;
    }
    el.gameList.innerHTML = rows.map(function (entry) {
      const key = entry[0];
      const frame = entry[1];
      const active = key === state.selectedGameKey ? " active" : "";
      const label = (frame.phase || "selfplay") +
        (frame.cycle ? " c" + frame.cycle : "") +
        " g" + (frame.game_index + 1);
      const sub = "ply " + (frame.ply || 0) + (frame.outcome ? " · " + frame.outcome : "");
      return "<button class=\"game-button" + active + "\" data-game=\"" + escapeHtml(key) + "\">" +
        "<span>" + escapeHtml(label) + "</span><span class=\"chip\">" + escapeHtml(frame.event) + "</span>" +
        "<span class=\"subtle\">" + escapeHtml(sub) + "</span><span></span>" +
        "</button>";
    }).join("");
    el.gameList.querySelectorAll("button[data-game]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.selectedGameKey = button.getAttribute("data-game");
        const frames = currentGameFrames();
        if (frames.length) {
          state.selectedIndex = state.frames.indexOf(frames[frames.length - 1]);
        }
        render();
      });
    });
  }

  function renderMoves() {
    const frames = currentGameFrames().filter(function (frame) {
      return Boolean(frame.move);
    });
    if (!frames.length) {
      el.moveList.innerHTML = "<li class=\"empty\">No moves yet.</li>";
      return;
    }
    const selected = currentFrame();
    el.moveList.innerHTML = frames.map(function (frame, index) {
      const active = frame === selected ? " active" : "";
      const selectedMark = frame.search && frame.search.selected_move_id === frame.move.id ? "chosen" : "move";
      return "<li><button class=\"move-button" + active + "\" data-frame=\"" + state.frames.indexOf(frame) + "\">" +
        "<span>" + index + ". " + escapeHtml(frame.move.label) + "</span><span class=\"chip\">" + selectedMark + "</span>" +
        "<span class=\"subtle\">" + escapeHtml(frame.move.piece_id) + "</span><span class=\"subtle\">" + escapeHtml(frame.move.short_id) + "</span>" +
        "</button></li>";
    }).join("");
    el.moveList.querySelectorAll("button[data-frame]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.selectedIndex = Number(button.getAttribute("data-frame"));
        render();
      });
    });
  }

  function renderSearch() {
    const frame = currentFrame();
    const search = frame ? frame.search : null;
    if (!search) {
      el.rootValue.textContent = "-";
      el.legalMoves.textContent = "-";
      el.searchList.innerHTML = "<li class=\"empty\">Search data appears after live self-play moves.</li>";
      return;
    }
    el.rootValue.textContent = formatNumber(search.root_value, 3);
    el.legalMoves.textContent = String(search.legal_move_count);
    if (!search.top_moves || !search.top_moves.length) {
      el.searchList.innerHTML = "<li class=\"empty\">No ranked moves.</li>";
      return;
    }
    el.searchList.innerHTML = search.top_moves.map(function (item) {
      const pct = Math.max(0, Math.min(100, Math.round((item.probability || 0) * 100)));
      const label = item.selected ? item.label + " selected" : item.label;
      return "<li class=\"search-item\">" +
        "<strong>" + escapeHtml(label) + "</strong>" +
        "<span class=\"subtle\">" + item.visits + " visits · " + pct + "% · " + escapeHtml(item.short_id) + "</span>" +
        "<span class=\"search-meter\"><span style=\"width:" + pct + "%\"></span></span>" +
        "</li>";
    }).join("");
  }

  function renderEvents() {
    const frame = currentFrame();
    const events = frame && frame.delta && Array.isArray(frame.delta.events) ? frame.delta.events : [];
    if (!events.length) {
      el.eventList.innerHTML = "<li class=\"empty\">No hook or engine events for this frame.</li>";
      return;
    }
    el.eventList.innerHTML = events.map(function (event) {
      const target = event.target_piece_id ? " -> " + event.target_piece_id : "";
      return "<li class=\"event-item\">" +
        "<strong>" + escapeHtml(event.event_type) + escapeHtml(target) + "</strong>" +
        "<span class=\"subtle\">actor " + escapeHtml(event.actor_piece_id || "-") + " · " + escapeHtml(event.source_cause || "-") + "</span>" +
        "</li>";
    }).join("");
  }

  function renderActivity() {
    const recent = state.frames.slice(-80).reverse();
    if (!recent.length) {
      el.activityList.innerHTML = "<li class=\"empty\">Waiting for viewer events.</li>";
      return;
    }
    el.activityList.innerHTML = recent.map(function (frame) {
      const metadataMessage = frame.metadata && frame.metadata.message ? " · " + frame.metadata.message : "";
      const game = frame.game_index !== null && frame.game_index !== undefined ? " g" + (frame.game_index + 1) : "";
      const ply = frame.ply !== null && frame.ply !== undefined ? " ply " + frame.ply : "";
      return "<li class=\"activity-item\">" +
        "<strong>" + escapeHtml(frame.event) + "</strong>" +
        "<span class=\"subtle\">" + escapeHtml((frame.phase || "viewer") + game + ply + metadataMessage) + "</span>" +
        "</li>";
    }).join("");
  }

  function startPlayback() {
    state.playing = true;
    el.playPause.textContent = "Pause";
    if (state.timer) {
      clearInterval(state.timer);
    }
    state.timer = setInterval(function () {
      const offset = selectedOffsetInGame();
      const frames = currentGameFrames();
      if (offset < 0 || offset >= frames.length - 1) {
        stopPlayback();
        return;
      }
      setSelectedFrameByGameOffset(offset + 1);
    }, state.speedMs);
  }

  function stopPlayback() {
    state.playing = false;
    el.playPause.textContent = "Play";
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  function bindControls() {
    el.stepBack.addEventListener("click", function () {
      stopPlayback();
      setSelectedFrameByGameOffset(selectedOffsetInGame() - 1);
    });
    el.stepForward.addEventListener("click", function () {
      stopPlayback();
      setSelectedFrameByGameOffset(selectedOffsetInGame() + 1);
    });
    el.playPause.addEventListener("click", function () {
      if (state.playing) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });
    el.jumpLive.addEventListener("click", function () {
      stopPlayback();
      for (let index = state.frames.length - 1; index >= 0; index -= 1) {
        const frame = state.frames[index];
        if (hasBoard(frame)) {
          state.selectedGameKey = gameKey(frame);
          state.selectedIndex = index;
          break;
        }
      }
      render();
    });
    el.flipBoard.addEventListener("click", function () {
      state.orientation = state.orientation === "white" ? "black" : "white";
      render();
    });
    el.speedRange.addEventListener("input", function () {
      state.speedMs = Number(el.speedRange.value);
      if (state.playing) {
        startPlayback();
      }
    });
  }

  function connectEvents() {
    const source = new EventSource("/events");
    source.onmessage = function (event) {
      addFrame(JSON.parse(event.data));
    };
    source.onerror = function () {
      addActivityOnly({
        event: "viewer_disconnected",
        phase: "viewer",
        metadata: { message: "event stream disconnected" }
      });
    };
  }

  function addActivityOnly(frame) {
    state.frames.push(Object.assign({
      cycle: null,
      game_index: null,
      games_total: null,
      ply: null,
      before: null,
      after: null,
      move: null,
      delta: null,
      search: null,
      outcome: null,
      termination_reason: null
    }, frame));
    render();
  }

  function loadReplayPayload() {
    return fetch("/api/replay")
      .then(function (response) {
        return response.json();
      })
      .then(function (payload) {
        if (payload && Array.isArray(payload.frames)) {
          payload.frames.forEach(addFrame);
        }
      });
  }

  function formatNumber(value, places) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "-";
    }
    return Number(value).toFixed(places);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  bindControls();
  render();
  loadReplayPayload().then(connectEvents).catch(function () {
    addActivityOnly({
      event: "viewer_load_failed",
      phase: "viewer",
      metadata: { message: "could not load replay payload" }
    });
    connectEvents();
  });
}());
