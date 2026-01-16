<script setup>
import { ref, reactive, nextTick } from 'vue';
import FinanceCard from './FinanceCard.vue';
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({
  html: true,        // 允许 HTML 标签
  linkify: true,     // 自动转换链接
  typographer: true,
});

// 1. 状态定义
const messages = ref([]);
const userInput = ref('');
const messagesContainer = ref(null); // 用于控制滚动条

const TYPING_SPEED = 30;


const isSending = ref(false);

// 2. 发送消息核心逻辑
const sendMessage = async () => {
  if (!userInput.value.trim() || isSending.value) return; // 如果正在发送，则拦截

  isSending.value = true;

  const text = userInput.value;
  userInput.value = '';

  // 添加用户消息
  messages.value.push({role: 'user', content: text});

  // 添加一个“空的”AI消息占位符（响应式对象）
  const currentAiMsg = reactive({
    role: 'assistant',
    content: '',
    _rawContent: '',
    sentiment: '',     // 稍后由 metadata 填充
    financeData: null, // 稍后由 finance 填充
    status: 'streaming'
  });
  messages.value.push(currentAiMsg);

  // 启动打字机定时器
  const typingTimer = setInterval(() => {
    if (currentAiMsg._rawContent.length > currentAiMsg.content.length) {
      // 从缓冲区取下一个字符
      currentAiMsg.content += currentAiMsg._rawContent.charAt(currentAiMsg.content.length);
      scrollToBottom();
    } else if (currentAiMsg.status === 'done') {
      // 如果后端发完了，且缓冲区也打完了，清空定时器
      clearInterval(typingTimer);
    }
  }, TYPING_SPEED);

  try {
    // 3. 发起请求
    const response = await fetch('http://127.0.0.1:8000/api/chat/stream', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' // 明确告诉后端我们要流
        },
        body: JSON.stringify({ message: text })
    });

    //console.log("响应状态码:", response.status);
    
    if (!response.ok) {
        const errText = await response.text();
        console.error("后端返回错误详情:", errText);
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 4. 读取流
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let pendingBuffer = "";

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      pendingBuffer = decoder.decode(value, { stream: true });

      let boundary = pendingBuffer.indexOf('\n\n');

      while (boundary !== -1){
        const completeBlock = pendingBuffer.slice(0, boundary).trim();
        pendingBuffer = pendingBuffer.slice(boundary + 2);
        
        if (completeBlock) {
          //console.log(">>> 收到原始块:", JSON.stringify(pendingBuffer));
            processSSEEvent(completeBlock, currentAiMsg);
        }

        boundary = pendingBuffer.indexOf('\n\n')
      }
    }
  } catch (error) {
    currentAiMsg._rawContent += "\n[网络连接错误]";
  } finally {
    currentAiMsg.status = 'done';
    isSending.value = false; // 无论成功失败，结束后释放按钮
  }
};

// 5. 解析 SSE 事件块
const processSSEEvent = (block, currentAiMsg) => {
    const lines = block.split('\n');
    let eventType = 'message'; // 默认事件类型
    let dataContent = '';

    lines.forEach(line => {
        if (line.startsWith('event:')) {
            eventType = line.replace('event:', '').trim();
        } else if (line.startsWith('data:')) {
            dataContent = line.replace('data:', '').trim();
        }
    });

    if (!dataContent) return;

    try {
        if (eventType === 'metadata') {
            const meta = JSON.parse(dataContent);
            currentAiMsg.sentiment = meta.sentiment;
            
        } 
        else if (eventType === 'finance') {
            const finance = JSON.parse(dataContent);
            currentAiMsg.financeData = finance;
            // 金融数据由 FinanceCard 渲染，不操作 _rawContent
        }
        else if (eventType === 'message') {
            // 只有文本消息进入打字机缓冲区
            currentAiMsg._rawContent += dataContent;
        } 
        else if (eventType === 'end') {
            currentAiMsg.status = 'done';
        }
    } catch (e) {
        // 如果是纯文本而不是 JSON，且解析失败，可以视作普通消息（容错处理）
        if (eventType === 'message') {
            currentAiMsg._rawContent += dataContent;
        }
        console.warn("SSE JSON Parse Error:", e);
    }
};
const scrollToBottom = async () => {
  await nextTick();
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
};
</script>


<template>
  <div class="chat-wrapper"> 
    
    <div class="chat-window">
      
      <div class="messages" ref="messagesContainer">
        <div v-for="(msg, index) in messages" :key="index" 
             class="message-row" :class="msg.role">
          
          <div v-if="msg.role === 'user'" class="bubble user">
            {{ msg.content }}
          </div>

          <div v-else class="ai-wrapper">
            <transition name="pop">
              <div v-if="msg.sentiment" class="sentiment-tag" :data-type="msg.sentiment">
                {{ msg.sentiment.toUpperCase() === 'POSITIVE' ? '📈 看多' : '📉 看空' }}
              </div>
            </transition>

            <transition name="slide">
              <FinanceCard v-if="msg.financeData" :data="msg.financeData" />
            </transition>

            <div class="bubble ai">
              <div v-html="md.render(msg.content || '')"></div>
              <span v-if="msg.status === 'streaming'" class="typing-cursor"></span>
            </div>
          </div>
        </div>
      </div>
      
      <div class="input-area">
        <input 
          v-model="userInput" 
          @keyup.enter="sendMessage" 
          :disabled="isSending" 
          placeholder="问问股价或财报分析..." 
        />
        <button 
          @click="sendMessage" 
          :disabled="isSending || !userInput.trim()" 
          :class="{ 'btn-loading': isSending }"
        >
          {{ isSending ? '思考中...' : '发送' }}
        </button>
      </div>
    </div> </div> 
</template>

<style scoped>
/* 容器布局 */
/* Markdown 内部样式微调 */
.markdown-body {
  font-size: 15px;
  line-height: 1.6;
  color: #2c3e50;
}

/* 让段落之间有间距 */
.markdown-body :deep(p) {
  margin-bottom: 10px;
}

/* 重点：加粗金融关键词 */
.markdown-body :deep(strong) {
  color: #e63946; /* 使用醒目的红色或深色 */
  font-weight: 700;
}

/* 让表格看起来像专业的研报表格 */
.markdown-body :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
  background: #fff;
}

.markdown-body :deep(th), 
.markdown-body :deep(td) {
  border: 1px solid #dfe2e5;
  padding: 8px 12px;
  text-align: left;
}

.markdown-body :deep(th) {
  background-color: #f6f8fa;
}

/* 列表样式 */
.markdown-body :deep(ul), 
.markdown-body :deep(ol) {
  padding-left: 20px;
  margin-bottom: 10px;
}
.chat-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100vw;
  height: 100vh;
  background: #f0f2f5;
}
.chat-window {
  width: 100%;
  max-width: 800px;
  height: 80vh;
  background: white;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  border-radius: 8px;
  overflow: hidden;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f9f9f9;
}

.input-area {
  padding: 15px;
  border-top: 1px solid #eee;
  display: flex;
  gap: 10px;
}
.citations-box {
  margin: 12px 0 8px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #4a90e2;
  font-size: 0.9em;
  color: #444;
}

.citations-title {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 8px;
}

.citation-item {
  margin: 8px 0;
  padding: 6px 0;
  border-bottom: 1px solid #eee;
}

.cite-index {
  color: #4a90e2;
  font-weight: bold;
  margin-right: 6px;
}

.file {
  font-weight: 500;
  color: #1e3a8a;
}

.score {
  margin-left: 8px;
  color: #6b7280;
  font-size: 0.85em;
}

.score.high {
  color: #16a34a;
  font-weight: bold;
}

.snippet {
  margin-top: 4px;
  color: #64748b;
  font-size: 0.85em;
  line-height: 1.4;
  font-style: italic;
}

input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

/* 消息气泡 */
.message-row {
  margin-bottom: 20px;
  display: flex;
}
.message-row.user { justify-content: flex-end; }
.message-row.assistant { justify-content: flex-start; }

.bubble {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.5;
  white-space: pre-wrap; /* 保留换行 */
}
.user .bubble {
  background: #007bff;
  color: white;
  border-bottom-right-radius: 2px;
}
.ai .bubble {
  background: white;
  border: 1px solid #e0e0e0;
  border-bottom-left-radius: 2px;
  color: #333;
}

/* 情感标签 */
.badge {
  display: inline-block;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  margin-bottom: 4px;
  font-weight: bold;
  color: white;
}
.badge.positive { background: #28a745; }
.badge.negative { background: #dc3545; }
.badge.neutral { background: #6c757d; }

/* 股票卡片样式 */
.stock-card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 8px;
  width: 200px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  animation: slideIn 0.3s ease-out;
}
.card-header {
  display: flex;
  justify-content: space-between;
  font-weight: bold;
  margin-bottom: 4px;
}
.change.up { color: #d32f2f; } /* A股红涨 */
.change.down { color: #2e7d32; } /* A股绿跌 */

/* 动画 */
@keyframes slideIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}
.cursor {
  display: inline-block;
  animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* 情感标签样式 */
.sentiment-tag {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: bold;
  margin-bottom: 8px;
}
.sentiment-tag[data-type="POSITIVE"] { background: #fee2e2; color: #ef4444; }
.sentiment-tag[data-type="NEGATIVE"] { background: #dcfce7; color: #22c55e; }

/* 消息动效 */
.pop-enter-active { animation: pop-in 0.3s ease-out; }
.slide-enter-active { animation: slide-in 0.4s ease-out; }

@keyframes pop-in {
  0% { transform: scale(0.8); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}

@keyframes slide-in {
  0% { transform: translateX(-20px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}

/* 打字机光标 */
.typing-cursor {
  display: inline-block;
  width: 2px;
  height: 15px;
  background: #007bff;
  margin-left: 4px;
  animation: blink 0.8s infinite;
}
</style>