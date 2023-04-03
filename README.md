# Burst Skinning for Unity

### Fast Mesh Skinning on the CPU using Linear Blend (LBS) and Dual Quaternion (DQS) utilizing the Unity Jobs system and Burst compilation
---
This repository is currently a WIP. The code is fully functional but not yet well documented, and may or may not be expanded upon in the future.

To get started, add a `BurstSkinner` component to a GameObject and configure it. Alternatively implement the interface `IBurstSkinnable` and pass deriving instances to `BurstSkinningUtility.Skin()` for more customizable skinning.

This code was developed and tested in Unity 2022.2+, older versions technically work but are not officially supported.
---
##### Dependencies (Unity Package Manager):
* Unity.Burst
* Unity.Collections
* Unity.Jobs
* Unity.Mathematics
---
##### References
* Ladislav Kavan - https://skinning.org/direct-methods.pdf