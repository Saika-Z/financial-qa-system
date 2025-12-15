<!--
 * @Author: Wenqing Zhao
 * @Date: 2025-12-15 21:32:05
 * @LastEditTime: 2025-12-15 21:45:23
 * @Description: 
 * @FilePath: /financial-qa-system/frontend/src/components/StockInfo.vue
-->
<template>
  <div>
    <h2>Stock Information</h2>
    <div>
      <input v-model="ticker" @keyup.enter="fetchStockData" placeholder="Enter stock ticker" />
      <button @click="fetchStockData">Get Stock Data</button>
    </div>
    <div v-if="stockData">
      <h3>Stock Data for: {{ stockData.symbol }}</h3>
      <p><strong>Current Price:</strong> {{ stockData.currentPrice }}</p>
      <p><strong>Market Cap:</strong> {{ stockData.marketCap }}</p>
      <p><strong>PE Ratio:</strong> {{ stockData.peRatio }}</p>
    </div>
    <div v-if="error" class="error">
      <p>{{ error }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      ticker: "AAPL", // 默认股票代码
      stockData: null,
      error: null,
    };
  },
  methods: {
    async fetchStockData() {
      this.error = null;
      this.stockData = null;

      try {
        const response = await fetch(`http://127.0.0.1:8000/finance/stock/${this.ticker}/info`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        this.stockData = {
          symbol: data.symbol,
          currentPrice: data.currentPrice,
          marketCap: data.marketCap,
          peRatio: data.peRatio,
        };
      } catch (err) {
      console.error('Fetch error:', err); 
        this.error = err.message;
      }
    }
  }
};
</script>

<style scoped>
.error {
  color: red;
}
</style>
