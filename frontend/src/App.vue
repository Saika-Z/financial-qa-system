<script setup>
import ChatContainer from './components/ChatContainer.vue';
</script>

<template>
  <div class="app-layout">
    <ChatContainer />
  </div>
</template>

<style>
body {
  margin: 0;
  background-color: #f4f4f9;
  font-family: 'Helvetica Neue', Arial, sans-serif;
}
.app-layout {
  display: flex;
  justify-content: center;
  height: 100vh;
  padding-top: 20px;
}
</style>