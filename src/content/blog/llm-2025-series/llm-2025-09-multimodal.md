---
title: '多模态大模型应用'
description: '视觉、语音、视频多模态 AI 应用开发，打造全方位智能体验。'
pubDate: 2025-07-20
category: '技术'
tags: ['多模态', '计算机视觉', '语音识别', 'LLM']
series: '2025 大模型实战'
seriesOrder: 9
heroImage: 'https://images.unsplash.com/photo-1535378917042-10a22c95931a?w=800&h=400&fit=crop'
---

多模态大模型能够同时理解文本、图像、音频和视频，为 AI 应用带来全新可能。本文将介绍主流多模态模型的使用方法和应用场景。

## 多模态能力概览

### 主流模型对比

| 模型 | 文本 | 图片 | 音频 | 视频 | 特点 |
|------|------|------|------|------|------|
| GPT-4o | ✅ | ✅ | ✅ | ✅ | 全能均衡 |
| Claude 3.5 | ✅ | ✅ | ❌ | ❌ | 图片理解强 |
| Gemini 2.0 | ✅ | ✅ | ✅ | ✅ | 视频最强 |
| Qwen-VL | ✅ | ✅ | ❌ | ❌ | 开源最强 |

### 应用场景

- **图片理解**: 图像描述、OCR、图表分析、商品识别
- **音频处理**: 语音转文字、会议记录、语音助手
- **视频分析**: 视频摘要、内容审核、教程讲解

## 图像理解应用

### GPT-4o 图片分析

```python
from openai import OpenAI
import base64

client = OpenAI()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def analyze_image(image_path: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# 使用示例
result = analyze_image("chart.png", "请分析这张图表，提取关键数据")
print(result)
```

### 多图对比

```python
def compare_images(images: list, prompt: str) -> str:
    content = [{"type": "text", "text": prompt}]
    
    for img_path in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
            }
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1500
    )
    return response.choices[0].message.content

# 对比两张图片
result = compare_images(
    ["before.jpg", "after.jpg"],
    "对比这两张图片，找出所有差异"
)
```

### OCR 文字提取

```python
def extract_text_from_image(image_path: str) -> str:
    prompt = """请提取图片中的所有文字内容。
要求：
1. 保持原有的格式和结构
2. 如果是表格，用 Markdown 表格格式输出
3. 如果有手写文字，尽量识别"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

# 提取发票信息
invoice_text = extract_text_from_image("invoice.jpg")
print(invoice_text)
```

## 语音处理应用

### Whisper 语音转文字

```python
from openai import OpenAI

client = OpenAI()

def transcribe_audio(audio_path: str, language: str = None) -> str:
    """语音转文字"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="text"
        )
    return transcript

def transcribe_with_timestamps(audio_path: str) -> dict:
    """带时间戳的转写"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript

# 使用
text = transcribe_audio("meeting.mp3", language="zh")
print(text)
```

### 文字转语音

```python
def text_to_speech(text: str, voice: str = "alloy", output_path: str = "output.mp3"):
    """文字转语音"""
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
        input=text
    )
    response.stream_to_file(output_path)
    return output_path

# 生成语音
text_to_speech("你好，欢迎使用 AI 语音服务！", voice="nova")
```

### 语音对话助手

```python
import tempfile

class VoiceAssistant:
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
    
    def listen(self, audio_path: str) -> str:
        """语音转文字"""
        with open(audio_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    
    def think(self, user_input: str) -> str:
        """生成回复"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一位友好的语音助手，回答要简洁。"},
                *self.conversation_history
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def speak(self, text: str) -> str:
        """文字转语音"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            response.stream_to_file(f.name)
            return f.name
    
    def process(self, audio_path: str) -> str:
        """完整处理流程"""
        user_text = self.listen(audio_path)
        print(f"用户: {user_text}")
        
        response_text = self.think(user_text)
        print(f"助手: {response_text}")
        
        audio_output = self.speak(response_text)
        return audio_output

# 使用
assistant = VoiceAssistant()
output_audio = assistant.process("user_question.mp3")
```

## 视频分析应用

### Gemini 视频理解

```python
import google.generativeai as genai
import time

genai.configure(api_key="your-api-key")

def analyze_video(video_path: str, prompt: str) -> str:
    """分析视频内容"""
    video_file = genai.upload_file(video_path)
    
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError("视频处理失败")
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([video_file, prompt])
    
    return response.text

# 使用示例
summary = analyze_video(
    "lecture.mp4",
    "请总结这段视频的主要内容，列出关键时间点和对应的主题"
)
print(summary)
```

### 视频问答

```python
class VideoQA:
    def __init__(self):
        genai.configure(api_key="your-api-key")
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.video_file = None
    
    def load_video(self, video_path: str):
        """加载视频"""
        self.video_file = genai.upload_file(video_path)
        while self.video_file.state.name == "PROCESSING":
            time.sleep(5)
            self.video_file = genai.get_file(self.video_file.name)
        print("视频加载完成")
    
    def ask(self, question: str) -> str:
        """对视频提问"""
        if not self.video_file:
            return "请先加载视频"
        
        response = self.model.generate_content([
            self.video_file,
            question
        ])
        return response.text

# 使用
qa = VideoQA()
qa.load_video("tutorial.mp4")
print(qa.ask("这个视频讲了什么？"))
print(qa.ask("视频中提到了哪些技术？"))
```

## 实战案例：智能文档分析

```python
import json

class DocumentAnalyzer:
    def __init__(self):
        self.client = OpenAI()
    
    def analyze_pdf_page(self, image_path: str) -> dict:
        """分析 PDF 页面"""
        prompt = """分析这个文档页面，返回 JSON 格式：
{
    "type": "类型（文本/表格/图表/混合）",
    "title": "标题（如有）",
    "content": "主要内容摘要",
    "tables": [],
    "figures": ["图表描述"],
    "key_points": ["关键信息点"]
}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def extract_table(self, image_path: str) -> str:
        """提取表格为 Markdown"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "将图片中的表格转换为 Markdown 格式"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content

# 使用
analyzer = DocumentAnalyzer()
result = analyzer.analyze_pdf_page("report_page.png")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 实战案例：会议记录助手

```python
class MeetingAssistant:
    def __init__(self):
        self.client = OpenAI()
    
    def transcribe_meeting(self, audio_path: str):
        """转写会议录音"""
        with open(audio_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        return transcript
    
    def generate_summary(self, transcript: str) -> dict:
        """生成会议摘要"""
        prompt = f"""根据以下会议记录生成结构化摘要：

{transcript}

请返回 JSON 格式包含：title, participants, key_decisions, action_items, summary"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一位专业的会议记录助手"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def process(self, audio_path: str) -> dict:
        """完整处理"""
        transcript = self.transcribe_meeting(audio_path)
        full_text = "\n".join([
            f"[{seg['start']:.1f}s] {seg['text']}" 
            for seg in transcript.segments
        ])
        summary = self.generate_summary(full_text)
        summary["transcript"] = full_text
        return summary

# 使用
assistant = MeetingAssistant()
result = assistant.process("meeting_recording.mp3")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 总结

多模态大模型应用的关键点：

1. **选择合适模型**: 
   - 图片理解: GPT-4o, Claude 3.5
   - 视频分析: Gemini 2.0
   - 语音处理: Whisper + TTS

2. **优化输入质量**:
   - 图片清晰度
   - 音频降噪
   - 视频压缩

3. **结构化输出**:
   - 使用 JSON 格式
   - 定义清晰的 Schema

4. **成本控制**:
   - 图片压缩
   - 视频分段处理
   - 缓存常见结果

下一篇将介绍企业级大模型部署与微调。
