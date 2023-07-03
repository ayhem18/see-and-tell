<div align="center">
<h1>
See and Tell
</h1>
<h4>
AI-driven Assistant to Experience Visual Content
</h4>
<h4>
  <a href="https://pytorch.org/">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>
  </a>
  <a href="https://github.com/teexone/see-and-tell/blob/a5bd742c1d4ff088c56f887cfc9c34bf58b7bc44/Dockerfile">
  <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"/>
  </a>
  <a href="https://huggingface.co/">
  <img style="height: 2em" src="https://huggingface.co/datasets/huggingface/badges/raw/main/powered-by-huggingface-light.svg">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=EVBhl29Ns0U">
  <img src="https://img.shields.io/badge/Demo-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white"/>
  </a>
  <a href="https://github.com/teexone/see-and-tell/">
  <img src="https://img.shields.io/badge/linkedin_Post-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>  
  <a href="https://github.com/teexone/see-and-tell/">
  <img src="https://img.shields.io/badge/Habr_Post-%2365A5B3.svg?style=for-the-badge&logo=habr&logoColor=white"/>
  </a> 
</h4>
</div>

# :dart: Goal

Our service aims to make visual content more accessible for individuals with visual impairments. We provide detailed audio descriptions of movies, TV shows, images, and more, allowing visually impaired users to fully experience and enjoy these media. Additionally, our solution caters to situations where active viewing is not possible, like when driving, providing an immersive audio experience instead. Our mission is to promote inclusivity and ensure that everyone, regardless of their visual abilities, can engage with and appreciate visual content.

# 💻 Service

<img src="https://i.imgur.com/32q0smh.png"/>

Our service operates through a streamlined pipeline consisting of five essential components. First, the **Describe** component utilizes an image-to-text model to generate textual descriptions of the events happening on the screen. Next, the **Listen** component intelligently identifies dialogue moments in the video to avoid overlapping with audio descriptions. The **Recognize** component employs face detection to identify characters, enhancing the context of captions by including character names. The **Say** component utilizes text-to-speech technology to voice the generated captions. Finally, **Mixer** combines the voiced captions with the original video, producing a final result video where the audio descriptions seamlessly blend with the visual content.

Most of components exploit <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1em"/>HuggingFace models, such are SpeechT5, GIT, Audio Segmentation and etc. TODO: WRITE ABOUT NLP AND FACE RECOGNITION

# 🚀 Demo

To provide a demo of our work, we took a 30-seconds fragment from _The Big Bang Theory_ TV series and processed it. 
<div align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=EVBhl29Ns0U
" target="_blank"><img src="https://i.imgur.com/KqqeErQ.png"/> 
</a>
</div>

The work done so far can be summarized as following:
- [+] Audio descriptions embedded without dialogues interventions
- [+] The voice is clear and nice
- [+] Captions are mostly correct and descriptive 
- [+] Characters are recognized correctly in most cases

However, after watching a demo you might notice one of the following:
- [!] Leonard was recognized as Sheldon, because Sheldon's face was more visible. However, the center figure in the frame was still Leonard. So, the service produced a caption 'Sheldon points to a brick wall' while actually it was Leonard who pointed.
- [!] At the very end, the model describe scene as 'A man in green jacket and red shirt ...', but the video was paused on different frame at that moment. The reason is that we describe every second, while frame rate of the source video is not divisible by seconds. That is why, the actual frame that was described as 'A man in ... ' is following the one the video was paused on immediately.

# 🛠️ Reproduce

To reproduce and run the service locally, you are encouraged to use Docker.

```bash
git clone https://github.com/teexone/see-and-tell/ seeandtell
cd seeandtell
docker build -t seeandtell .
docker run --rm -v /path/to/video/folder:/video seeandtell python -m cntell --help
```

# License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
