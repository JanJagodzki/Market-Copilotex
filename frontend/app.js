const searchInput = document.querySelector("#symbol-search");
const searchButton = document.querySelector("#search-button");
const searchResults = document.querySelector("#search-results");
const resultCount = document.querySelector("#result-count");
const refreshButton = document.querySelector("#refresh-button");
const chartElement = document.querySelector("#chart");
const chartMessage = document.querySelector("#chart-message");

let currentTicker = "AAPL";
let chart;
let candleSeries;


function createPriceChart() {
    chart = LightweightCharts.createChart(chartElement, {
        layout: {
            background: { type: "solid", color: "#10151d" },
            textColor: "#8995a5",
        },
        grid: {
            vertLines: { color: "#1b2430" },
            horzLines: { color: "#1b2430" },
        },
        rightPriceScale: {
            borderColor: "#26303d",
        },
        timeScale: {
            borderColor: "#26303d",
            timeVisible: true,
            secondsVisible: false,
        },
        crosshair: {
            vertLine: { color: "#657184" },
            horzLine: { color: "#657184" },
        },
    });

    candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: "#20c997",
        downColor: "#ff6470",
        wickUpColor: "#20c997",
        wickDownColor: "#ff6470",
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
    changeElement.textContent = `${difference >= 0 ? "+" : ""}${difference.toFixed(2)} (${percent.toFixed(2)}%) in visible range`;
    changeElement.className = difference >= 0 ? "positive" : "negative";
}


async function loadSymbol(ticker) {
    currentTicker = ticker.toUpperCase();
    showChartMessage(`Loading ${currentTicker}…`);

    try {
        const response = await fetch(
            `/api/symbols/${encodeURIComponent(currentTicker)}/prices?interval=15m&limit=200`
        );

        if (!response.ok) {
            throw new Error("Price data is not available");
        }

        const data = await response.json();

        document.querySelector("#symbol-ticker").textContent = data.ticker;
        document.querySelector("#symbol-name").textContent = data.name || "Company name unavailable";

        if (data.prices.length === 0) {
            showChartMessage("No 15-minute candles for this company yet");
            return;
        }

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

        document.querySelectorAll(".symbol-result").forEach((button) => {
            button.classList.toggle("active", button.dataset.ticker === currentTicker);
        });
    } catch (error) {
        showChartMessage(error.message);
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
        const response = await fetch(`/api/symbols?search=${encodeURIComponent(search)}&limit=20`);

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
refreshButton.addEventListener("click", () => loadSymbol(currentTicker));

createPriceChart();
searchInput.value = "AAPL";
searchSymbols();
loadSymbol("AAPL");
