<template>
  <div class="sentiment-container">
    <h2>AI sentiment analyzing - </h2>
    <div class="input-group">
      <input v-model="question" placeholder="ask qistion to get postive, negative or neutral..." />
      <button @click="askAI" :disabled="isAnalyzing">
      {{ isAnalyzing ? 'Analyzing ...' : 'submit' }}
      </button>
    </div>
    <div v-if="aiAnswer" class="result-box">
      <strong>sentiment:</strong>
      {{ aiAnswer }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';


const question = ref('');
const aiAnswer = ref('');
const isAnalyzing = ref(false);


const askAI = async () => {

if (!question.value) return alert("Please ask a question.");

  isAnalyzing.value = true;
  
  try {
    const payload = {
      question: question.value
    };

    // console.log(payload);
    console.log("Sending Payload:", JSON.stringify(payload));

    const res = await axios.post(
    'http://127.0.0.1:8000/sentiment/query', 
    payload,
   { headers: { 'Content-Type': 'application/json' } }
   );
    aiAnswer.value = res.data.answer || res.data.result || res.data;
    
    // console.log(res);
    console.log(res.data.answer);

  } catch (e) {
    aiAnswer.value = "AI is busy now, please try later.";
  }finally{
   isAnalyzing.value=false;
  }
};
</script>