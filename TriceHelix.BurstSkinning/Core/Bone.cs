using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace TriceHelix.BurstSkinning.Core
{
    public struct Bone
    {
        // transformation
        public float4x4 worldToBindpose;
        public float4x4 bindposeToSkinned;
        public DualQuaternion bindposeToSkinnedDQ;

        // bone segment
        public float3 boundA;
        public float3 boundB;
        public float3 transformedA;
        public float3 transformedAB;
        public float dotTransformedAB;


        public float3 ProjectOnBone(float3 point)
        {
            return transformedA + (math.dot(point - transformedA, transformedAB) / dotTransformedAB * transformedAB);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DistToBone(float3 point, out float3 projection)
        {
            projection = ProjectOnBone(point);
            return math.length(point - projection);
        }
    }


    [BurstCompile]
    public struct ComputeVertexToBoneDistancesParallelJob : IJobParallelFor
    {
        // input
        public NativeArray<Bone>.ReadOnly bones;
        public NativeArray<BoneWeight1>.ReadOnly weights;
        public NativeArray<int>.ReadOnly weightsPerVertexScan;
        public StridedData<float3> vertices;

        // output
        [WriteOnly] public NativeArray<float> distances;


        // EXECUTE
        public void Execute(int index)
        {
            distances[index] = bones[weights[weightsPerVertexScan[index]].boneIndex].DistToBone(vertices[index], out _);
        }
    }


    [BurstCompile]
    public unsafe struct TransformBonesParallelJob : IJobParallelFor
    {
        // input
        public SkinningMethod skinningMethod;
        public bool enableBulgeOptimization;
        public float4x4 worldToRoot;
        [ReadOnly] public NativeArray<float4x4> boneToWorldTFs;

        // I/O
        [NativeMatchesParallelForLength]
        public NativeArray<Bone> bones;


        // EXECUTE
        public void Execute(int index)
        {
            // avoid local bone copy using direct ref to memory
            ref Bone b = ref UnsafeUtility.AsRef<Bone>((Bone*)bones.GetUnsafePtr() + index);

            // bindpose to skinned transform
            float4x4 tf = math.mul(math.mul(worldToRoot, boneToWorldTFs[index]), b.worldToBindpose);
            if (skinningMethod == SkinningMethod.LBS)
            {
                // store regular transformation matrix
                b.bindposeToSkinned = tf;
            }
            else if (skinningMethod == SkinningMethod.DQS)
            {
                // convert matrix to dual quaternion
                b.bindposeToSkinnedDQ = new DualQuaternion(new quaternion(tf), tf.c3.xyz);

                if (enableBulgeOptimization)
                {
                    // metadata used for DQS bulge optimization
                    float3 a = math.transform(tf, b.boundA);
                    float3 ab = math.transform(tf, b.boundB) - a;
                    b.transformedA = a;
                    b.transformedAB = ab;
                    b.dotTransformedAB = math.dot(ab, ab);
                }
            }
        }
    }
}
