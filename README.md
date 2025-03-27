# Generative AI

# Day 1 - GenAI, Algorithms, Architecture and Applications
This repository contains the basic concepts of Generative AI and its projects. 
## 🤖 Generative AI (GenAI)

Generative AI (GenAI) refers to a type of artificial intelligence that can create new content, such as text, images, audio, video, and code, based on patterns learned from existing data. It uses machine learning models, particularly deep learning architectures like Generative Adversarial Networks (GANs) 🎨, Variational Autoencoders (VAEs) 📊, and Transformer-based models (e.g., GPT, BERT, DALL·E) 📝, to generate human-like outputs.

GenAI is widely used in applications like:
- **Text Generation** (ChatGPT) 📝
- **Image Synthesis** (DALL·E, MidJourney) 🎨
- **Voice Cloning & Speech Synthesis** (Bark AI, Tortoise TTS) 🎤
- **Drug Discovery** 💊

It enables automation, personalization, and creativity across industries like entertainment 🎬, healthcare 🏥, and finance 💰.

## 🚀 Applications of GenAI
- **Text Generation** (ChatGPT, Bard) 📝
- **Image Synthesis** (DALL·E, MidJourney) 🎨
- **Voice Cloning & Speech Synthesis** (Bark AI, Tortoise TTS) 🎤
- **Music Composition** 🎵
- **Drug Discovery & Healthcare** 💊🏥
- **Automated Code Generation** (GitHub Copilot) 💻
- **Finance & Fraud Detection** 💰


## 🔬 Types of Generative AI Algorithms
### 1️⃣ Generative Adversarial Networks (GANs)
GANs consist of two neural networks:
- **Generator 🏗️**: Creates fake data resembling real data.
- **Discriminator 🛡️**: Evaluates whether the generated data is real or fake.
Both networks train together in a competitive process, improving each other's performance.

### 2️⃣ Variational Autoencoders (VAEs)
VAEs work with an encoder-decoder architecture:
- **Encoder 🔍**: Compresses input data into a latent space representation.
- **Decoder 🎨**: Reconstructs data from the latent space.
They use probabilistic methods to generate diverse and smooth outputs.

#### 🏗️ Architecture of GANs
1. **Generator:**
   - Takes random noise as input (e.g., Gaussian noise).
   - Uses deep neural networks (fully connected layers, CNNs, or transformers).
   - Outputs fake data that resembles real data.
2. **Discriminator:**
   - Takes real and fake data as input.
   - Classifies input as real or fake (binary classification).
   - Sends feedback to the generator to improve its fake outputs.

**Training Process:**
- The generator tries to fool the discriminator by producing realistic samples.
- The discriminator learns to distinguish real from fake samples.
- This adversarial training continues until the generator produces highly realistic outputs.

### 2️⃣ Variational Autoencoders (VAEs)
VAEs work with an encoder-decoder architecture:
- **Encoder 🔍**: Compresses input data into a latent space representation.
- **Decoder 🎨**: Reconstructs data from the latent space.
They use probabilistic methods to generate diverse and smooth outputs.

#### 🔍 Architecture of VAEs
1. **Encoder:**
   - Compresses input data into a lower-dimensional latent space.
   - Outputs a probability distribution (mean and variance) instead of a single value.
2. **Latent Space:**
   - Represents data in a compressed probabilistic form.
   - Enables controlled sampling for diverse data generation.
3. **Decoder:**
   - Reconstructs data from the latent space representation.
   - Uses deep neural networks (e.g., CNNs or fully connected layers).

**Training Process:**
- The encoder maps input to a latent distribution.
- A sample is drawn from the latent distribution.
- The decoder reconstructs the input from this sample.
- The loss function (reconstruction loss + KL divergence) ensures meaningful latent space representations.

## 🔍 Differences Between GANs and VAEs
| Feature         | GANs 🎨 | VAEs 📊 |
|---------------|--------|--------|
| Architecture  | Uses a Generator & Discriminator | Uses an Encoder & Decoder |
| Training      | Adversarial (Competitive) | Variational (Probabilistic) |
| Output Quality | High-Quality, Realistic | Smooth, but May Be Blurry |
| Stability     | Harder to Train | More Stable |
| Applications  | Image Generation, Deepfake | Data Reconstruction, Anomaly Detection |


## 🤖 Reinforcement Learning from Human Feedback (RLHF)
RLHF is used to fine-tune AI models using human preferences. It helps:
- Improve model safety and alignment with user expectations.
- Reduce biases in AI-generated outputs.
- Make responses more engaging and human-like (e.g., ChatGPT fine-tuning).
  
---

## Day 2 - Neural Networks, Types of Neural Networks, Use of Neural Networks in GenAI
## 🔬 Neural Networks 
Neural networks are the backbone of deep learning and Generative AI. They are computational models inspired by the human brain, designed to recognize patterns and make intelligent decisions.

## Types of Neural Networks
### 1️⃣ Feedforward Neural Networks (FNN)
- The simplest type of neural network.
- Information moves in one direction (input → hidden layers → output).
- Used for classification and regression tasks.

### 2️⃣ Convolutional Neural Networks (CNN)
- Primarily used for image processing.
- Consists of convolutional layers that extract features from input images.
- Used in applications like object detection, image synthesis, and facial recognition.

### 3️⃣ Recurrent Neural Networks (RNN)
- Designed for sequential data processing.
- Uses loops to retain information from previous steps (memory-like behavior).
- Applied in tasks like speech recognition, text generation, and time-series forecasting.

## 🔥 Why Use Neural Networks in GenAI?
- **Pattern Recognition**: They learn patterns from vast datasets to generate realistic outputs.
- **Creativity**: Enables AI to create new content like text, images, and music.
- **Data Efficiency**: Helps process large-scale structured and unstructured data.
- **Real-time Adaptation**: Used in applications like AI chatbots and recommendation systems.

---
