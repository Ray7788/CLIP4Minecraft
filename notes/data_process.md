Data and training details
===
Data preparation
---
This section provides detailed information about the dataset, covering its construction process and a sample of the dataset format. The steps involved in constructing the dataset are outlined below:

#### Here is an original method:
1. Obtain YouTube videos and corresponding transcripts from the MineDojo database.
2. Manually compile a list of keywords related to Minecraft gameplay.
3. Annotate all keywords, including various forms such as combined words and plural forms, found in the transcripts of each video.
4. Slide a window of length L words across the transcript until the first keyword in the window is about to exit. Use the midpoint between the first and last keywords in the window as the center to extract a transcript clip of length L words.
5. Extract all non-overlapping transcript clips from each video with a transcript following step 4.
6. For each transcript clip, calculate the central timestamp based on the transcript timestamps. Utilize this central timestamp to extract a video clip of duration D seconds from the corresponding video.
7. From all the extracted video-clip pairs, select M pairs and encode them using a pre-trained variant of MineCLIP attention to compute the cosine similarity.
8. Choose the top k% pairs with the highest cosine similarity as the training set from the M pairs.
9. Additionally, randomly select M0 pairs along with the M pairs to form the validation set.

The parameters used in the above process are detailed in Table 6. Steps 2-6 constitute content filtering, while steps 7-8 represent correlation filtering. Following this methodology, a training set of size 640K and a test set of size 4096 are constructed. Both content and correlation filtering methods are applied to construct the training set, whereas only content filtering is applied to the test set to mirror the data distribution in the database.

additional keywords:
https://www.minecraftforum.net/forums/show-your-creation/video-series-help/technical-help/1715074-popular-youtube-tags-for-minecraft-videos

#### New added dataset method：
Based on the skill information already available, extract the name of each skill as a search term keyword and add the words Minecraft at the end of the keyword to pick out videos with subtitles that are in the top 10 in terms of plays, and then continue to repeat the steps of clipping the video-text pairs starting from step 4.

Data Processing
----
The script automates the process of downloading specific segments of videos from YouTube based on the information provided in a JSON file. It uses the yt_dlp YoutubeDL library to download a specific segment of the video from YouTube based on the provided start and end times.

The overall logic is to sample 16 frames from an MP4 video, preprocess each frame and convert it to a tensor, stack the 16 frames, and save them as a .pth file.

The initial video clipping function uniformly samples a certain number of frames (default is 16) from a given video file and returns them as a list of numpy arrays. Firstly, it gets the total number of frames in the video, calculates the indices of the frames to be sampled, and iterates over these indices, then it resizes the frame to the specified size, and adds it to the list of sampled frames. Secondly, it converts the list of sampled frames to a numpy array and returns it. Later, another function iterates over the list of frames, converts each frame to a tensor, and adds it to the list of preprocessed frames. Finally, it stacks the list of preprocessed frames into a single tensor and returns it.

Training details
---
The training procedure for CLIP4MC was adapted from the training methodologies of CLIP4MC and MineCLIP. Specifically, all models were trained using the 640K training dataset. For each video-text clip pair, we acquired 16 frames of RGB images via equidistant sampling and normalized each channel individually. Throughout the training process, temporally-consistent random resized cropping were employed for data augmentation. We utilized cosine learning rate annealing with a warm-up period consisting of 320 gradient steps. Fine-tuning was limited to the last two layers of pre-trained CLIP encoders, with module-wise learning rate decay implemented for improved fine-tuning (where the learning rate decays along with the modules). 

Additionally, we applied a lower learning rate (×0.5) on the pre-trained weights and implemented layer-wise learning rate decay to facilitate better fine-tuning.

Training was conducted on a single node equipped with 4 × V100 GPUs, employing FP16 mixed precision via the PyTorch native amp module.

* Hyperparameter Value
LR schedule Cosine with warmup
Warmup steps 320
LR 1.5e-4
Weight decay 0.2
Layerwise LR decay 0.65
Batch size per GPU 100
Parallel GPUs 4
Video resolution 160 × 256
Number of frames 16
Image encoder ViT-B/16

* LR schedule Cosine with warmup [73]
Warmup steps 500
Peak LR 1.5e-4
Final LR 1e-5
Weight decay 0.2
Layerwise LR decay 0.65
Pre-trained layers LR multiplier 0.5×
Batch size per GPU 64
Parallel GPUs 8
Video resolution 160 × 256
Number of frames 16
Image encoder ViT-B/16 [28]