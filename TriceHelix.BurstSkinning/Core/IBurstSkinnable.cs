using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;

namespace TriceHelix.BurstSkinning.Core
{
    /// <summary>
    /// Implement this interface on objects you wish to skin using <see cref="BurstSkinningUtility"/>.
    /// </summary>
    public interface IBurstSkinnable
    {
        /// <summary>
        /// Skinning Method/Algorithm (Linear Blend or Dual Quaternion)
        /// </summary>
        SkinningMethod GetSkinningMethod();

        /// <summary>
        /// Amount of bulge optimization applied when <see cref="SkinningMethod.DQS"/> is selected, in range [0;1]
        /// </summary>
        float GetBulgeOptimizationFactor();

        /// <summary>
        /// Enable/Disable the skinning of vertex normals
        /// </summary>
        bool GetEnableNormalSkinning();

        /// <summary>
        /// Bones of the rig
        /// </summary>
        NativeArray<Bone> GetBones();

        /// <summary>
        /// Bone local-to-world transformations
        /// </summary>
        NativeArray<float4x4> GetBoneTransforms();

        /// <summary>
        /// World to root bone transformation
        /// </summary>
        /// <returns></returns>
        float4x4 GetWorldToRoot();

        /// <summary>
        /// Equivalent of <see cref="Mesh.GetAllBoneWeights()"/>
        /// </summary>
        NativeArray<BoneWeight1>.ReadOnly GetWeights();

        /// <summary>
        /// Equivalent of <see cref="Mesh.GetBonesPerVertex"/>
        /// </summary>
        NativeArray<byte>.ReadOnly GetWeightsPerVertex();

        /// <summary>
        /// Exclusive scan of <see cref="GetWeightsPerVertex"/>
        /// </summary>
        NativeArray<int>.ReadOnly GetWeightsPerVertexScan();

        /// <summary>
        /// Value of <see cref="Bone.DistToBone(float3, out float3)"/> for every vertex in its bindpose, only required when <see cref="SkinningMethod.DQS"/> is selected
        /// </summary>
        NativeArray<float>.ReadOnly GetVertexDistancesToBone();

        /// <summary>
        /// Enable/Disable post-skinning bounds computation
        /// </summary>
        bool GetEnableBoundsCalc();

        /// <summary>
        /// Reference to the bounds of the skinned mesh, only required if <see cref="GetEnableBoundsCalc"/> is true
        /// </summary>
        NativeReference<Bounds> GetBoundsRef();

        // INPUT/OUTPUT (Get<...>Normals() does not need to be implemented when GetEnableNormalSkinning() returns false)
        int GetVertexCount();
        StridedData<float3> GetInputVertices();
        StridedData<float3> GetInputNormals();
        StridedData<float3> GetOutputVertices();
        StridedData<float3> GetOutputNormals();
    }


    public enum SkinningMethod
    {
        LBS,
        DQS
    }
}
