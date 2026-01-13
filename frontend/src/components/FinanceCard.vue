<template>
  <div class="finance-card">
    <div class="card-header">
      <div class="stock-info">
        <span class="symbol">{{ data.symbol }}</span>
        <span class="name">{{ data.name || '股票插件' }}</span>
      </div>
      <div class="status-dot" :class="isUp ? 'up' : 'down'"></div>
    </div>
    
    <div class="card-body">
      <div class="price-section">
        <div class="current-price">${{ data.price }}</div>
        <div class="price-change" :class="isUp ? 'up' : 'down'">
          {{ isUp ? '▲' : '▼' }} {{ data.change_percent }}
        </div>
      </div>
      
      <div class="mini-chart">
        <div class="chart-bar" v-for="i in 8" :key="i" 
             :style="{ height: Math.random() * 100 + '%', opacity: 0.3 + (i/10) }"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
const props = defineProps(['data']);
const isUp = computed(() => !props.data.change_percent.includes('-'));
</script>

<style scoped>
.finance-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  border-left: 4px solid #333;
  border-radius: 8px;
  padding: 16px;
  width: 260px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.08);
  margin: 10px 0;
  border: 1px solid #eaeaea;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.symbol {
  font-size: 1.2rem;
  font-weight: 800;
  color: #1a1a1a;
  margin-right: 8px;
}

.name {
  font-size: 0.8rem;
  color: #666;
}

.current-price {
  font-size: 1.5rem;
  font-weight: 700;
  color: #222;
}

.price-change {
  font-size: 0.9rem;
  font-weight: 600;
  margin-top: 4px;
}

.up { color: #eb4432; border-color: #eb4432; } /* 金融红跌绿涨可根据习惯调整 */
.down { color: #15ad5c; border-color: #15ad5c; }

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

.mini-chart {
  display: flex;
  align-items: flex-end;
  gap: 3px;
  height: 30px;
  margin-top: 15px;
}
.chart-bar {
  flex: 1;
  background: currentColor;
  border-radius: 2px;
}
</style>