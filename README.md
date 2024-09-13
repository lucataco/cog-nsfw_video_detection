# Falconsai/nsfw_video_detection Cog model

This is an implementation of [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) as a [Cog](https://github.com/replicate/cog) model extended to work with video files.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction

    cog predict -i image=@falcon.mp4

![input](falcon.gif)

# Output

The model will either return the string "normal" or "nsfw"
