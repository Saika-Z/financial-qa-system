<template>
  <div id="app">
    <nav class="navbar">
      <button @click="currentPage = 'finance'">股票行情</button>
      <button @click="currentPage = 'sentiment'">情感分析</button>
    </nav>
    <div class="content">
      <StockFinance v-if="currentPage === 'finance'" :ticker="currentTicker" />
      <SentimentAnalysis v-else-if="currentPage === 'sentiment'" :ticker="currentTicker" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import StockFinance from './components/StockFinance.vue'
import SentimentAnalysis from './components/SentimentAnalysis.vue';

const currentPage = ref('finance');
const currentTicker = ref('AAPL');
</script>

<style>
.navbar { margin-bottom: 20px; display: flex; gap: 10px; }
.navbar button { padding: 10px 20px; cursor: pointer; }
</style>
