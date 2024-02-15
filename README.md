# trajectory_analysis
## Small Target Recognition Based on Trajectory Information

Recognizing small targets has always posed a significant challenge in image analysis. Over long distances, the image signal-to-noise ratio tends to be low, limiting the amount of useful information available to detection systems. Consequently, visual target recognition becomes an intricate task to tackle. In this study, we introduce an innovative Track Before Detect (TBD) approach that leverages target trajectory information (coordinates) to effectively distinguish between noise and potential targets.
By reframing the problem as a multivariate time series classification, we have achieved remarkable results. Specifically, our TBD method achieves an impressive 97% accuracy in separating target signals from noise within a mere half-second time span (consisting of 10 data points). Furthermore, when classifying the identified targets into our predefined categories—airplane, drone, and bird—we achieve an outstanding classification accuracy of 96% over a more extended period of 1.5 seconds (comprising 30 data points).

## Resources

- Sample videos can be found on the link https://www.youtube.com/playlist?list=PLvPINJRuTlZ8hc7ZCZxFf55Cf2_1X6D-s
- dataset files can be found on the link https://www.kaggle.com/datasets/saadkentar/track-with-noise-v2


## How to use

- Use train.py to train the model on your trajectory dataset
- Pretrained weights for both binary and multicalss classification are provided with the code. namely "target_classes_model_30p_90.keras" and "target_noise_model_v2_10p_96.keras"
- Use main.py to test the proposed algorithm on videos
- sample videos can be found on https://github.com/DroneDetectionThesis/Drone-detection-dataset