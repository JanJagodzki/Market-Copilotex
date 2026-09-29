const searchInput = document.querySelector("#symbol-search");
const searchButton = document.querySelector("#search-button");
const searchResults = document.querySelector("#search-results");
const resultCount = document.querySelector("#result-count");
const watchlistResults = document.querySelector("#watchlist-results");
const watchlistCount = document.querySelector("#watchlist-count");
const watchlistButton = document.querySelector("#watchlist-button");
const refreshButton = document.querySelector("#refresh-button");
const chartElement = document.querySelector("#chart");
const chartMessage = document.querySelector("#chart-message");
const horizonList = document.querySelector("#horizon-list");
const modelStatus = document.querySelector("#model-status");
const syncStatus = document.querySelector("#sync-status");
const tradeForm = document.querySelector("#trade-form");
const tradeSide = document.querySelector("#trade-side");
const tradeQuantity = document.querySelector("#trade-quantity");
const tradePrice = document.querySelector("#trade-price");
const tradeNote = document.querySelector("#trade-note");
const tradeSubmit = document.querySelector("#trade-submit");
const tradeMessage = document.querySelector("#trade-message");
const tradeHistory = document.querySelector("#trade-history");

let currentTicker = "AAPL";
let watchlistTickers = new Set();
let chart;
let candleSeries;


function createPriceChart() {
    chart = LightweightCharts.createChart(chartElement, {
        layout: {
            background: { type: "solid", color: "#ffffff" },
            textColor: "#444444",
        },
        grid: {
            vertLines: { color: "#eeeeee" },
            horzLines: { color: "#eeeeee" },
        },
        rightPriceScale: {
            borderColor: "#bbbbbb",
        },
        timeScale: {
            borderColor: "#bbbbbb",
            timeVisible: true,
            secondsVisible: false,
        },
        crosshair: {
            vertLine: { color: "#777777" },
            horzLine: { color: "#777777" },
        },
    });

    candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: "#666666",
        downColor: "#111111",
        wickUpColor: "#666666",
        wickDownColor: "#111111",
        borderVisible: false,
    });

    const observer = new ResizeObserver((entries) => {
        const width = entries[0].contentRect.width;
        chart.resize(width, chartElement.clientHeight);
    });

    observer.observe(chartElement);
}


function showChartMessage(message) {
    chartElement.style.display = "none";
    chartMessage.style.display = "grid";
    chartMessage.textContent = message;
}


function showChart() {
    chartMessage.style.display = "none";
    chartElement.style.display = "block";
}


function updatePriceSummary(prices) {
    const latest = prices[prices.length - 1];
    const first = prices[0];
    const difference = latest.close - first.open;
    const percent = (difference / first.open) * 100;
    const changeElement = document.querySelector("#price-change");

    document.querySelector("#last-price").textContent = `$${latest.close.toFixed(2)}`;
    tradePrice.value = latest.close.toFixed(2);
    changeElement.textContent = `${difference >= 0 ? "+" : ""}${difference.toFixed(2)} (${percent.toFixed(2)}%) in visible range`;
    changeElement.className = difference >= 0 ? "positive" : "negative";
}


function formatMoney(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    return `$${Number(value).toFixed(2)}`;
}


function renderPaperTrades(data) {
    const summary = data.summary;
    const profitElement = document.querySelector("#portfolio-profit");

    document.querySelector("#portfolio-caption").textContent = (
        `${data.ticker} transaction history`
    );
    document.querySelector("#portfolio-shares").textContent = summary.shares;
    document.querySelector("#portfolio-average").textContent = formatMoney(
        summary.average_buy_price
    );
    document.querySelector("#portfolio-value").textContent = formatMoney(
        summary.market_value
    );
    profitElement.textContent = formatMoney(summary.total_profit_loss);
    profitElement.className = summary.total_profit_loss >= 0
        ? "positive"
        : "negative";

    tradeHistory.replaceChildren();

    if (data.trades.length === 0) {
        tradeHistory.textContent = "No paper trades yet";
        return;
    }

    data.trades.forEach((trade) => {
        const row = document.createElement("div");
        const details = document.createElement("div");
        const title = document.createElement("strong");
        const date = document.createElement("span");
        const note = document.createElement("small");
        const removeButton = document.createElement("button");

        row.className = "trade-row";
        title.className = trade.side === "BUY" ? "positive" : "negative";
        title.textContent = (
            `${trade.side} ${trade.quantity} × ${formatMoney(trade.price)}`
        );
        date.textContent = new Date(trade.traded_at).toLocaleString();
        note.textContent = trade.note || "No note";
        removeButton.type = "button";
        removeButton.textContent = "Delete";
        removeButton.addEventListener("click", () => deletePaperTrade(trade.id));

        details.append(title, date, note);
        row.append(details, removeButton);
        tradeHistory.append(row);
    });
}


async function loadPaperTrades(ticker) {
    try {
        const response = await fetch(
            `/api/paper-trades/${encodeURIComponent(ticker)}`
        );

        if (!response.ok) {
            throw new Error("Paper portfolio is not available");
        }

        renderPaperTrades(await response.json());
    } catch (error) {
        tradeHistory.textContent = error.message;
    }
}


async function savePaperTrade(event) {
    event.preventDefault();
    tradeSubmit.disabled = true;
    tradeMessage.textContent = "Saving…";

    const payload = {
        side: tradeSide.value,
        quantity: Number(tradeQuantity.value),
        price: Number(tradePrice.value),
        note: tradeNote.value,
    };

    try {
        const response = await fetch(
            `/api/paper-trades/${encodeURIComponent(currentTicker)}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }
        );

        if (!response.ok) {
            const error = await response.json();
            throw new Error(
                typeof error.detail === "string"
                    ? error.detail
                    : "Trade data is invalid"
            );
        }

        renderPaperTrades(await response.json());
        tradeQuantity.value = "";
        tradeNote.value = "";
        tradeMessage.textContent = "Paper trade saved";
    } catch (error) {
        tradeMessage.textContent = error.message;
    } finally {
        tradeSubmit.disabled = false;
    }
}


async function deletePaperTrade(tradeId) {
    if (!window.confirm("Delete this paper trade?")) {
        return;
    }

    const response = await fetch(
        `/api/paper-trades/trade/${tradeId}`,
        { method: "DELETE" }
    );

    if (!response.ok) {
        const error = await response.json();
        window.alert(error.detail || "Could not delete the trade");
        return;
    }

    await loadPaperTrades(currentTicker);
}


async function getSymbolPrices(ticker) {
    const response = await fetch(
        `/api/symbols/${encodeURIComponent(ticker)}/prices?interval=15m&limit=200`
    );

    if (!response.ok) {
        throw new Error("Price data is not available");
    }

    return response.json();
}


async function downloadSymbolPrices(ticker) {
    const response = await fetch(
        `/api/symbols/${encodeURIComponent(ticker)}/prices/sync?days=5`,
        { method: "POST" }
    );

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Price download failed");
    }

    return response.json();
}


function renderPriceData(data) {
    document.querySelector("#symbol-ticker").textContent = data.ticker;
    document.querySelector("#symbol-name").textContent = data.name || "Company name unavailable";

    const candles = data.prices.map((price) => ({
        time: Math.floor(new Date(price.timestamp).getTime() / 1000),
        open: price.open,
        high: price.high,
        low: price.low,
        close: price.close,
    }));

    candleSeries.setData(candles);
    chart.timeScale().fitContent();
    updatePriceSummary(data.prices);
    document.querySelector("#last-update").textContent = `Loaded ${new Date().toLocaleTimeString()}`;
    showChart();

    document.querySelectorAll("[data-ticker]").forEach((element) => {
        element.classList.toggle(
            "active",
            element.dataset.ticker === currentTicker
        );
    });
}


async function loadPriceData(ticker, downloadIfEmpty = true) {
    let data = await getSymbolPrices(ticker);

    if (data.prices.length === 0 && downloadIfEmpty) {
        showChartMessage(`Downloading ${ticker} candles…`);
        await downloadSymbolPrices(ticker);
        data = await getSymbolPrices(ticker);
    }

    if (data.prices.length === 0) {
        throw new Error("Yahoo returned no 15-minute candles");
    }

    renderPriceData(data);
}


function renderPredictions(data) {
    horizonList.replaceChildren();

    data.predictions.forEach((prediction) => {
        const item = document.createElement("div");
        const horizon = document.createElement("span");
        const probability = document.createElement("strong");
        const details = document.createElement("small");

        item.className = "horizon-item";
        horizon.textContent = `${prediction.horizon_days}D`;
        const directionProbability = prediction.direction === "up"
            ? prediction.probability_up
            : 1 - prediction.probability_up;

        probability.textContent = `${(directionProbability * 100).toFixed(1)}% ${prediction.direction}`;
        probability.className = prediction.direction === "up" ? "positive" : "negative";
        details.textContent = `${prediction.model} · ${prediction.quality}`;

        item.append(horizon, probability, details);
        horizonList.append(item);
    });

    modelStatus.textContent = `Data ${data.data_date}`;

    const best = data.predictions.reduce((current, prediction) => (
        prediction.validation_auc > current.validation_auc
            ? prediction
            : current
    ));

    const directionProbability = best.direction === "up"
        ? best.probability_up
        : 1 - best.probability_up;

    const probability = (directionProbability * 100).toFixed(1);
    const direction = best.direction === "up" ? "growth" : "decline";

    document.querySelector("#ai-title").textContent = (
        `${probability}% probability of ${direction} in ${best.horizon_days} trading days`
    );

    document.querySelector("#ai-description").textContent = (
        `The strongest evaluated model is ${best.model} with test AUC ${best.test_auc.toFixed(4)}. `
        + "Short horizons are weaker and should not be treated as buy or sell signals."
    );
}


async function loadPredictions(ticker) {
    modelStatus.textContent = "Loading";

    try {
        const response = await fetch(
            `/api/symbols/${encodeURIComponent(ticker)}/ai-predictions`
        );

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Predictions are not available");
        }

        renderPredictions(await response.json());
    } catch (error) {
        modelStatus.textContent = "Unavailable";
        horizonList.textContent = error.message;
        document.querySelector("#ai-title").textContent = "AI analysis unavailable";
        document.querySelector("#ai-description").textContent = error.message;
    }
}


function updateWatchlistButton() {
    const isAdded = watchlistTickers.has(currentTicker);

    watchlistButton.textContent = isAdded
        ? "Remove from watchlist"
        : "Add to watchlist";

    watchlistButton.classList.toggle(
        "active",
        isAdded
    );
}


function renderWatchlist(items) {
    watchlistResults.replaceChildren();
    watchlistCount.textContent = items.length;
    watchlistTickers = new Set(
        items.map((item) => item.ticker)
    );

    if (items.length === 0) {
        watchlistResults.textContent = "No watched companies";
        updateWatchlistButton();
        return;
    }

    items.forEach((item) => {
        const row = document.createElement("div");
        const openButton = document.createElement("button");
        const ticker = document.createElement("strong");
        const removeButton = document.createElement("button");

        row.className = "watchlist-row";
        openButton.type = "button";
        openButton.className = "watchlist-open";
        openButton.dataset.ticker = item.ticker;
        ticker.textContent = item.ticker;
        removeButton.type = "button";
        removeButton.className = "watchlist-remove";
        removeButton.textContent = "×";
        removeButton.title = `Remove ${item.ticker}`;

        openButton.append(ticker);
        openButton.addEventListener("click", () => loadSymbol(item.ticker));
        removeButton.addEventListener("click", () => removeFromWatchlist(item.ticker));
        row.append(openButton, removeButton);
        watchlistResults.append(row);
    });

    updateWatchlistButton();
}


async function loadWatchlist() {
    try {
        const response = await fetch("/api/watchlist");

        if (!response.ok) {
            throw new Error("Watchlist is not available");
        }

        renderWatchlist(await response.json());
    } catch (error) {
        watchlistResults.textContent = error.message;
    }
}


async function addToWatchlist(ticker) {
    const response = await fetch(
        `/api/watchlist/${encodeURIComponent(ticker)}`,
        { method: "POST" }
    );

    if (!response.ok) {
        throw new Error("Could not add the company");
    }

    await loadWatchlist();
}


async function removeFromWatchlist(ticker) {
    const response = await fetch(
        `/api/watchlist/${encodeURIComponent(ticker)}`,
        { method: "DELETE" }
    );

    if (!response.ok) {
        throw new Error("Could not remove the company");
    }

    await loadWatchlist();
}


async function toggleCurrentWatchlist() {
    watchlistButton.disabled = true;

    try {
        if (watchlistTickers.has(currentTicker)) {
            await removeFromWatchlist(currentTicker);
        } else {
            await addToWatchlist(currentTicker);
        }
    } catch (error) {
        window.alert(error.message);
    } finally {
        watchlistButton.disabled = false;
    }
}


async function loadSyncStatus() {
    try {
        const response = await fetch("/api/watchlist/status/current");

        if (!response.ok) {
            throw new Error();
        }

        const status = await response.json();

        if (status.running) {
            syncStatus.textContent = "Updating watchlist prices";
        } else if (status.last_finished) {
            const time = new Date(status.last_finished).toLocaleTimeString();
            syncStatus.textContent = `Auto sync: ${status.message} at ${time}`;
        } else {
            syncStatus.textContent = `${status.message} · every ${status.interval_minutes} min`;
        }
    } catch (error) {
        syncStatus.textContent = "Auto sync unavailable";
    }
}


async function loadSymbol(ticker) {
    currentTicker = ticker.toUpperCase();
    showChartMessage(`Loading ${currentTicker}…`);
    updateWatchlistButton();

    try {
        await loadPriceData(currentTicker);
    } catch (error) {
        showChartMessage(error.message);
    }

    await loadPredictions(currentTicker);
    await loadPaperTrades(currentTicker);
}


async function refreshCurrentSymbol() {
    refreshButton.disabled = true;
    refreshButton.textContent = "Updating…";

    try {
        await downloadSymbolPrices(currentTicker);
        await loadPriceData(currentTicker, false);
        await loadPredictions(currentTicker);
        await loadPaperTrades(currentTicker);
    } catch (error) {
        showChartMessage(error.message);
    } finally {
        refreshButton.disabled = false;
        refreshButton.textContent = "Refresh";
    }
}


function renderSearchResults(symbols) {
    searchResults.replaceChildren();
    resultCount.textContent = symbols.length;

    if (symbols.length === 0) {
        searchResults.textContent = "No companies found";
        return;
    }

    symbols.forEach((symbol) => {
        const button = document.createElement("button");
        const ticker = document.createElement("strong");
        const name = document.createElement("span");

        button.type = "button";
        button.className = "symbol-result";
        button.dataset.ticker = symbol.ticker;
        ticker.textContent = symbol.ticker;
        name.textContent = symbol.name || "Unknown company";

        button.append(ticker, name);
        button.addEventListener("click", () => loadSymbol(symbol.ticker));
        searchResults.append(button);
    });
}


async function searchSymbols() {
    const search = searchInput.value.trim();

    try {
        const response = await fetch(
            `/api/symbols?search=${encodeURIComponent(search)}&limit=20`
        );

        if (!response.ok) {
            throw new Error("Search failed");
        }

        renderSearchResults(await response.json());
    } catch (error) {
        searchResults.textContent = error.message;
        resultCount.textContent = "0";
    }
}


searchButton.addEventListener("click", searchSymbols);
searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        searchSymbols();
    }
});

refreshButton.addEventListener("click", refreshCurrentSymbol);
watchlistButton.addEventListener("click", toggleCurrentWatchlist);
tradeForm.addEventListener("submit", savePaperTrade);

createPriceChart();
searchInput.value = "AAPL";
searchSymbols();
loadWatchlist();
loadSyncStatus();
loadSymbol("AAPL");

setInterval(() => {
    loadSyncStatus();

    if (!document.hidden) {
        loadPriceData(currentTicker, false).catch(() => {});
    }
}, 60 * 1000);
