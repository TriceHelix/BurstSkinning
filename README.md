# Burst Skinning for Unity

### Fast Mesh Skinning on the CPU using Linear Blend (LBS) and Dual Quaternion (DQS) utilizing the Unity Jobs system and Burst compilation
Requires Unity 2022.2 and newer.

---

#### Installation (via Package Manager)
* Click "Add package from git URL..."
* Enter `https://github.com/TriceHelix/BurstSkinning.git` and click "Add"
* Done!

*NOTE: This will install the following dependencies:*
* Unity.Burst
* Unity.Collections
* Unity.Mathematics

---

#### Getting Started
Add a `BurstSkinner` component to a GameObject and configure it. (Add Component -> Mesh -> Burst Skinner)

Alternatively you can implement the interface `IBurstSkinnable` and pass deriving instances to `BurstSkinningUtility.Skin()` for more customizable skinning.

This repository is still a major WIP and is currently used internally for a larger project. More Documentation will be added in the future.

---

##### References
* Ladislav Kavan - https://skinning.org/direct-methods.pdf