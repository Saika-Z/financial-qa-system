<!--
 * @Author: Wenqing Zhao
 * @Date: 2025-12-15 21:30:30
 * @LastEditTime: 2025-12-15 21:39:38
 * @Description: 
 * @FilePath: /financial-qa-system/frontend/src/App.vue
-->
<template>
  <div id="app">
    <StockInfo />
  </div>
</template>

<script>
import StockInfo from './components/StockInfo.vue'

export default {
  name: 'App',
  components: {
    StockInfo
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
