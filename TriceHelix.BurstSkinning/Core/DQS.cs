using System;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace TriceHelix.BurstSkinning.Core
{
    public static unsafe partial class SkinningImpl
    {
        public const float DQS_BULGE_OPTIMIZATION_THRESHOLD = 0.001f;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DQS(
            bool skinNormal,
            float optimizationFactor,
            ref float3 vertex,
            ref float3 normal,
            in Span<BoneWeight1> weights,
            in NativeArray<Bone>.ReadOnly bones,
            float orgDistToBone)
        {
            // dual quaternion transformation from original to skinned
            DualQuaternion dq = default;
            for (int i = 0; i < weights.Length; i++)
                dq += weights[i].weight * UnsafeUtility.ArrayElementAsRef<Bone>(bones.GetUnsafeReadOnlyPtr(), weights[i].boneIndex).bindposeToSkinnedDQ;

            dq = dq.Normalized();
            vertex = dq.Transform(vertex);
            if (skinNormal) normal = math.rotate(dq.real, normal);

            if (optimizationFactor >= DQS_BULGE_OPTIMIZATION_THRESHOLD)
            {
                // project skinned vertex onto highest weighted bone
                float skDistToBone = UnsafeUtility.ArrayElementAsRef<Bone>(bones.GetUnsafeReadOnlyPtr(), 0).DistToBone(vertex, out float3 skBoneProj);

                // pull bulging vertices back to bone
                if (skDistToBone > orgDistToBone)
                    vertex = math.lerp(vertex, skBoneProj + (orgDistToBone / skDistToBone * (vertex - skBoneProj)), optimizationFactor);
            }
        }
    }


    [BurstCompile]
    public unsafe struct DualQuaternionSkinningParallelJob : IJobParallelFor
    {
        // input
        public bool enableNormalSkinning;
        public float optimizationFactor;
        public StridedData<float3> originalVertices;
        public StridedData<float3> originalNormals;
        public NativeArray<Bone>.ReadOnly bones;
        public NativeArray<float>.ReadOnly orgDistancesToBone;
        public NativeArray<BoneWeight1>.ReadOnly boneWeights;
        public NativeArray<byte>.ReadOnly weightsPerVertex;
        public NativeArray<int>.ReadOnly weightsPerVertexScan;

        // output
        public StridedData<float3> skinnedVertices;
        public StridedData<float3> skinnedNormals;


        // EXECUTE
        public void Execute(int index)
        {
            // bone weights
            int numWeights = weightsPerVertex[index];
            if (numWeights == 0) return;
            BoneWeight1* ptr_weights = stackalloc BoneWeight1[numWeights];
            UnsafeUtility.MemCpy(ptr_weights, (BoneWeight1*)boneWeights.GetUnsafeReadOnlyPtr() + weightsPerVertexScan[index], numWeights * sizeof(BoneWeight1));

            // get vertex and normal
            float3 vertex = originalVertices[index];
            float3 normal = enableNormalSkinning ? originalNormals[index] : default;

            SkinningImpl.DQS(
                enableNormalSkinning,
                optimizationFactor,
                ref vertex,
                ref normal,
                new Span<BoneWeight1>(ptr_weights, numWeights),
                bones,
                orgDistancesToBone[index]);

            // set vertex and normal
            skinnedVertices[index] = vertex;
            if (enableNormalSkinning) skinnedNormals[index] = normal;
        }
    }
}
