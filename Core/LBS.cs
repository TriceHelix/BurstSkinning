using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using System;

namespace TriceHelix.BurstSkinning.Core
{
    public static unsafe partial class SkinningImpl
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void LBS(
            bool skinNormal,
            ref float3 vertex,
            ref float3 normal,
            in Span<BoneWeight1> weights,
            in NativeArray<Bone>.ReadOnly bones)
        {
            // transformation matrix from original to skinned
            float4x4 tf = default;
            for (int i = 0; i < weights.Length; i++)
                tf += weights[i].weight * UnsafeUtility.ArrayElementAsRef<Bone>(bones.GetUnsafeReadOnlyPtr(), weights[i].boneIndex).bindposeToSkinned;

            vertex = math.transform(tf, vertex);
            if (skinNormal) normal = math.rotate(tf, normal);
        }
    }


    [BurstCompile]
    public unsafe struct LinearBlendSkinningParallelJob : IJobParallelFor
    {
        // input
        public bool enableNormalSkinning;
        public StridedData<float3> originalVertices;
        public StridedData<float3> originalNormals;
        public NativeArray<Bone>.ReadOnly bones;
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

            SkinningImpl.LBS(
                enableNormalSkinning,
                ref vertex,
                ref normal,
                new Span<BoneWeight1>(ptr_weights, numWeights),
                bones);

            // set vertex and normal
            skinnedVertices[index] = vertex;
            if (enableNormalSkinning) skinnedNormals[index] = normal;
        }
    }
}
