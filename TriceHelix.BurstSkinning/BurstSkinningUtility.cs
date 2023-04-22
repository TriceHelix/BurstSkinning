using System;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using TriceHelix.BurstSkinning.Core;

namespace TriceHelix.BurstSkinning
{
    public static class BurstSkinningUtility
    {
        public static JobHandle Skin(IBurstSkinnable[] objects, JobHandle dependency = default)
        {
            NativeArray<JobHandle> handles = new(objects.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < objects.Length; i++)
            {
                IBurstSkinnable target = objects[i];
                handles[i] = Skin(target, dependency);
            }

            JobHandle combined = JobHandle.CombineDependencies(handles);
            handles.Dispose();
            return combined;
        }


        public static JobHandle Skin(IBurstSkinnable target, JobHandle dependency = default)
        {
            if (target == null)
                return dependency;

            // collect shared input data
            SkinningMethod method = target.GetSkinningMethod();
            float bulgeOptFactor = method == SkinningMethod.DQS ? target.GetBulgeOptimizationFactor() : 0f;
            bool skinNormals = target.GetEnableNormalSkinning();
            int vertCount = target.GetVertexCount();
            var iVert = target.GetInputVertices();
            var oVert = target.GetOutputVertices();
            var iNorm = skinNormals ? target.GetInputNormals() : default;
            var oNorm = skinNormals ? target.GetOutputNormals() : default;
            var bones = target.GetBones();
            var boneTransforms = target.GetBoneTransforms();
            var boneWeights = target.GetWeights();
            var weightsPerVertex = target.GetWeightsPerVertex();
            var weightsPerVertexScan = target.GetWeightsPerVertexScan();

            // some simple assertions
            Debug.Assert(vertCount > 0);
            Debug.Assert(bones.Length > 0);
            Debug.Assert(boneTransforms.Length == bones.Length);
            Debug.Assert(boneWeights.Length >= vertCount);
            Debug.Assert(weightsPerVertex.Length >= vertCount);
            Debug.Assert(weightsPerVertexScan.Length >= vertCount);

            // create and schedule transformation job
            var j1 = new TransformBonesParallelJob()
            {
                skinningMethod = method,
                enableBulgeOptimization = method == SkinningMethod.DQS && bulgeOptFactor >= SkinningImpl.DQS_BULGE_OPTIMIZATION_THRESHOLD,
                bones = bones,
                boneToWorldTFs = boneTransforms,
                worldToRoot = target.GetWorldToRoot()
            };
            JobHandle jh1 = j1.ScheduleByRef(bones.Length, 32, dependency);

            // create and schedule skinning job
            JobHandle jh2;
            if (method == SkinningMethod.LBS)
            {
                var j2 = new LinearBlendSkinningParallelJob()
                {
                    enableNormalSkinning = skinNormals,
                    bones = bones.AsReadOnly(),
                    boneWeights = boneWeights,
                    weightsPerVertex = weightsPerVertex,
                    weightsPerVertexScan = weightsPerVertexScan,
                    originalVertices = iVert,
                    originalNormals = iNorm,
                    skinnedVertices = oVert,
                    skinnedNormals = oNorm
                };
                jh2 = j2.ScheduleByRef(vertCount, 64, jh1);
            }
            else if (method == SkinningMethod.DQS)
            {
                var j2 = new DualQuaternionSkinningParallelJob()
                {
                    enableNormalSkinning = skinNormals,
                    optimizationFactor = target.GetBulgeOptimizationFactor(),
                    orgDistancesToBone = target.GetVertexDistancesToBone(),
                    bones = bones.AsReadOnly(),
                    boneWeights = boneWeights,
                    weightsPerVertex = weightsPerVertex,
                    weightsPerVertexScan = weightsPerVertexScan,
                    originalVertices = iVert,
                    originalNormals = iNorm,
                    skinnedVertices = oVert,
                    skinnedNormals = oNorm
                };
                jh2 = j2.ScheduleByRef(vertCount, 64, jh1);
            }
            else
            {
                throw new Exception($"Invalid Skinning Method: {method}");
            }

            // create and schedule bounds calculation job
            JobHandle jh3;
            if (target.GetEnableBoundsCalc())
            {
                var j3 = new ComputeBoundsJob()
                {
                    vertexCount = vertCount,
                    vertices = oVert,
                    ref_bounds = target.GetBoundsRef()
                };
                jh3 = j3.ScheduleByRef(jh2);
            }
            else
            {
                jh3 = jh2;
            }

            return jh3;
        }
    }
}
