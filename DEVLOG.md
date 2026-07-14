# Development Log

## 2026/07/13

### Progress
I set up the project environment and recorded ten golf swings at an outdoor driving range. However, I decided to record those swings again because I forgot to use slow-motion mode. Without slow-motion footage, it was difficult to capture clear images of the backswing top and impact phases because those moments occur very quickly.

I also gained a small blister on my left thumb while recording.

### Collecting Dataset
Because the weather was too hot, I decided to record the remaining twenty golf swings indoor after re-recording the ten outdor swings.

For the indoor swings, I plan to wear different outfits to increase the visual diversity of the datase. I'm not sure how much this will improve the model's generalization, but it should reduce the change that the model reies too heavily on one specific outfit.

### Environment Setup
I encountered several compatibility problems while installing PyTorch on my local machine.

I use an Intel-based MacBook Pro, and recent PyTorch versions no longer provide prebuilt packages for Intel macOS. Therefore, I created a Python 3.11 environment and installed versions compatible with PyTorch 2.2.2.

Current environment:
- python: 3.11.1
- numpy: 1.26.4
- pytorch: 2.2.2
- torchvision: 0.17.2


## 2026/07/14

### Progress
I re-recorded ten golf swings using slow-motion video. From each recording, I captured one image for each swing phase: address, top, impact, and finish. I then stored the images in their corresponding class directories.

This resulted in 40 images:
- 10 address images
- 10 backswing-top images
- 10 impact images
- 10 finish images

### Natural Variation in the Dataset
Because each swing was recorded separately, the extracted images are slightly different. The exact frame selected for each phase varies sightly between swings. The outdoor background also changes because of differences in sunlight, cloud movement, and the wind.

These small variations may help prevent the CNN from memorizing one identical image or background.