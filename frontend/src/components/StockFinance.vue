
<script setup>
import { ref , onMounted} from 'vue';
import axios from 'axios';

const props = defineProps(['ticker']);
const stockData = ref(null);
const loading = ref(true);
const error = ref(null);

const fetchAllData = async () => {
  loading.value = true;
  error.value = null;
  try {
    // this path must match the path in FastAPI register
    const response = await axios.get(`http://localhost:8000/finance/stock/${props.ticker}/all`);
    stockData.value = response.data;

    //
    //console.log("完整的股票数据:", stockData.value);



  } catch (e) {
    console.error("Error fetching data:", e);
    error.value = e.response?.status === 429
      ? "Rate limit exceeded, please try later."
      : "An error occurred while fetching data.";
  }finally {
    loading.value = false;
  }
};
onMounted(() => {
  fetchAllData();
});

</script>

<template>
  <div v-if="loading">Loading...</div>
  <div v-else-if="error">{{ error }}</div>
  <div v-else>
    <h1>{{ stockData.symbol }} -  ${{ stockData.price }}</h1>

    <StockChart :chartData = "stockData.history" />

    <ul>
      <li v-for="item in stockData.news" :key="item.uuid">
        {{item.title}}
      </li>
    </ul>
  </div>
</template>