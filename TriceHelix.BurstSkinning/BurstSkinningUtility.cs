using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using TriceHelix.BurstSkinning.Core;

namespace TriceHelix.BurstSkinning
{
    public static class BurstSkinningUtility
    {
        // cached allocations for performant single-alloc rig extraction for many skeletons
        private static readonly Stack<Transform> _ExtractRig_boneStack = new(512);
        private static readonly Stack<int> _ExtractRig_childIndexStack = new(512);
        private static readonly List<Transform> _ExtractRig_foundBones = new(512);
        private static readonly List<Transform> _ExtractRig_finalBones = new(512);

        /// <summary>
        /// Match mesh bindposes with their Transform counterpart.
        /// </summary>
        /// <param name="reference">Equvivalent to the transform of the <see cref="BurstSkinner"/> GameObject</param>
        /// <param name="root">Transform of the Root Bone</param>
        /// <param name="bindposes">Mesh bindposes (<see cref="Mesh.bindposes"/> or <see cref="Mesh.GetBindposes"/>)</param>
        /// <returns>Equivalent of <see cref="SkinnedMeshRenderer.bones"/></returns>
        public static Transform[] ExtractRig(Transform reference, Transform root, IEnumerable<Matrix4x4> bindposes)
        {
            if (reference == null || root == null || bindposes == null)
            {
                Debug.LogError($"One or more of these parameters are null:\n{nameof(reference)}: {reference}\n{nameof(root)}: {root}\n{nameof(bindposes)}: {bindposes}");
                return null;
            }

            _ExtractRig_boneStack.Clear();
            _ExtractRig_childIndexStack.Clear();
            _ExtractRig_foundBones.Clear();
            _ExtractRig_boneStack.Push(root.GetChild(0)); // true root is child of root bone
            _ExtractRig_childIndexStack.Push(0);

            // flatten bone hierarchy
            while (_ExtractRig_boneStack.TryPeek(out Transform b))
            {
                int cid = _ExtractRig_childIndexStack.Pop();

                if (cid == 0)
                    _ExtractRig_foundBones.Add(b);

                if (cid < b.childCount)
                {
                    // down
                    _ExtractRig_boneStack.Push(b.GetChild(cid));
                    _ExtractRig_childIndexStack.Push(cid + 1);
                    _ExtractRig_childIndexStack.Push(0);
                }
                else
                {
                    // up
                    _ExtractRig_boneStack.Pop();
                }
            }

            if (_ExtractRig_foundBones.Count < bindposes.Count())
            {
                Debug.LogError("Could not match every bindpose of the mesh to a bone!\nPlease ensure the root bone is set correctly and the mesh is properly rigged.");
                return null;
            }

            // filter out excess bones
            int specIdx = 0;
            var referenceTF = reference.worldToLocalMatrix;
            var realBindpositions = bindposes.Select(bp => bp.inverse.MultiplyPoint3x4(Vector3.zero)).ToArray();
            var specBindpositions = _ExtractRig_foundBones.Select(t => (referenceTF * t.localToWorldMatrix).MultiplyPoint3x4(Vector3.zero)).ToArray();
            _ExtractRig_finalBones.Clear();
            for (int i = 0; i < realBindpositions.Length; i++)
            {
                // find the closest matching transform down the hierarchy
                float delta = float.MaxValue;
                float prevDelta;
                for (int j = specIdx; j < specBindpositions.Length; j++)
                {
                    prevDelta = delta;
                    delta = (specBindpositions[j] - realBindpositions[i]).magnitude;
                    if (delta >= prevDelta)
                    {
                        specIdx = j - 1;
                        break;
                    }
                }

                _ExtractRig_finalBones.Add(_ExtractRig_foundBones[specIdx++]);
            }

            return _ExtractRig_finalBones.ToArray();
        }


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
