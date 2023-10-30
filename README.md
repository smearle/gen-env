# Untitled Game Language

- To evolve games for maximum complexity according to a search-based player agent, run `python evo_env.py`.
- To render the fittest environment-solution pairs found so far, run `python evo_env.py evaluate=True`
- To aggregate all unique environments that have been generated over the course of evolution, and record playtraces of their solutions as generated by search, run `python evo_env.py collect_elites=True`
- To imitation learn on these "oracle" playtraces, run `python train_il_player.py`.

To render in blender:
```bash
blender render_scene.blend --python enjoy_blender.py
```