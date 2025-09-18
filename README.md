# FLUX.1 Kontext Dev - Replicate Cog

A Replicate Cog wrapper for [FLUX.1 Kontext Dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), a 12 billion parameter rectified flow transformer capable of editing images based on text instructions.

## About FLUX.1 Kontext

FLUX.1 Kontext [dev] is an advanced image editing model that can modify existing images based on natural language instructions. It excels at making precise edits while maintaining consistency and quality.

### Key Features

- **Instruction-based editing**: Change existing images based on text descriptions
- **Character and style consistency**: Maintain reference without finetuning
- **Robust consistency**: Minimal visual drift through multiple successive edits
- **Guidance distillation**: More efficient generation process
- **High-quality output**: 12B parameter transformer for detailed results

## Usage

### Input Parameters

- **`prompt`** (string): Text instruction describing the edit to make
- **`image`** (file): Input image to edit
- **`aspect_ratio`** (string): Aspect ratio of the generated image
  - Options: `1:1`, `16:9`, `21:9`, `3:2`, `2:3`, `4:5`, `5:4`, `3:4`, `4:3`, `9:16`, `9:21`, `match_input_image`
  - Default: `match_input_image`
- **`guidance_scale`** (float): Guidance scale for generation (1.0 - 10.0)
  - Higher values follow the prompt more closely
  - Default: `2.5`
- **`num_inference_steps`** (int): Number of inference steps (1 - 50)
  - Default: `28`
- **`seed`** (int): Random seed for reproducibility
  - Set to `-1` for random seed
  - Default: `-1`

### Example Prompts

- "Add sunglasses to the person"
- "Change the background to a beach scene"
- "Make the car red instead of blue"
- "Add a hat to the cat"
- "Remove the person from the image"
- "Change day to night"

### Supported Aspect Ratios

The model supports various aspect ratios optimized for different use cases:

- **Square**: `1:1` (1024×1024)
- **Widescreen**: `16:9` (1344×768), `21:9` (1536×640)
- **Portrait**: `2:3` (832×1216), `3:4` (896×1152), `9:16` (768×1344)
- **Landscape**: `3:2` (1216×832), `4:3` (1152×896), `4:5` (944×1104), `5:4` (1104×944)
- **Ultra-wide**: `9:21` (640×1536)

## Content Safety

This implementation includes the PixtralContentFilter integrity checker to help prevent generation of inappropriate content. Images that are flagged by the content filter will raise an error with instructions to try a different prompt.

## Model Information

- **Model**: FLUX.1 Kontext [dev]
- **Parameters**: 12 billion
- **Architecture**: Rectified flow transformer
- **License**: FLUX.1 [dev] Non-Commercial License
- **Paper**: [arXiv:2506.15742](https://arxiv.org/abs/2506.15742)

## API Deployment

This model is available on Replicate at:
- [black-forest-labs/flux-kontext-dev](https://replicate.com/black-forest-labs/flux-kontext-dev)

## License

This model falls under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md). Generated outputs can be used for personal, scientific, and commercial purposes as described in the license.

## Citation

```bibtex
@misc{labs2025flux1kontextflowmatching,
      title={FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space}, 
      author={Black Forest Labs and Stephen Batifol and Andreas Blattmann and Frederic Boesel and Saksham Consul and Cyril Diagne and Tim Dockhorn and Jack English and Zion English and Patrick Esser and Sumith Kulal and Kyle Lacey and Yam Levi and Cheng Li and Dominik Lorenz and Jonas Müller and Dustin Podell and Robin Rombach and Harry Saini and Axel Sauer and Luke Smith},
      year={2025},
      eprint={2506.15742},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.15742},
}
```
