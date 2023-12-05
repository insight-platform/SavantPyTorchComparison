# SavantPyTorchComparison
Compare Savant and PyT

Common pipeline inference parameters:

- 640x640 inference dimensions
- 1 batch size
- fp16 mode

## Test video

The test 1442 frame h264 video used in the test can be downloaded by running

```bash
make get-test-video
```
## Models

## Results

Test | FPS
--- | ---
Naive Pytorch | 75
HW Decode Pytorch | 107
Savant | 255
