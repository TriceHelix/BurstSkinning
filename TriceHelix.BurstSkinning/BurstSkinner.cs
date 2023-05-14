using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using TriceHelix.BurstSkinning.Core;

namespace TriceHelix.BurstSkinning
{
    [AddComponentMenu("Mesh/Burst Skinner")]
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer)), DisallowMultipleComponent]
    public unsafe class BurstSkinner : MonoBehaviour, IBurstSkinnable
    {
        private const MeshUpdateFlags NO_VALIDATION_UPDATE_FLAGS =
                  MeshUpdateFlags.DontRecalculateBounds
                | MeshUpdateFlags.DontValidateIndices
                | MeshUpdateFlags.DontResetBoneBounds
                | MeshUpdateFlags.DontNotifyMeshUsers;

        #region INSPECTOR
        [SerializeField, Tooltip("Original mesh in bindpose (T-Pose)")]
        private Mesh sharedMesh = null;

        [SerializeField, Tooltip("Root bone of the skeleton/rig")]
        private Transform rootBone = null;

        [SerializeField, Tooltip("Method/Algorithm used for skinning")]
        private SkinningMethod method = SkinningMethod.LBS;

        [SerializeField, Range(0f, 1f), Tooltip("Amount of optimization applied after DQS to compensate for excessive bulging")]
        private float bulgeOptimizationFactor = 0f;

        [SerializeField, Tooltip("Update vertices and mesh bounds even when the renderer is invisible?")]
        private bool alwaysUpdate = false;
        #endregion

        // INTERNAL DATA
        private MeshFilter mf = null;
        private MeshRenderer mr = null;
        private VertexAttributeDescriptor[] outputVertexAttributes = Array.Empty<VertexAttributeDescriptor>();
        private bool enableNormalSkinning = false;
        private int outputVertexBufferStride = 0;
        private int outputIndexCount = 0;
        private Mesh outputMesh = null;
        private Mesh.MeshDataArray outputMDA = default;
        private NativeReference<Bounds> outputBounds = default;
        private Transform[] boneTransforms = Array.Empty<Transform>();
        private NativeArray<Bone> bones = default;
        private Mesh.MeshDataArray inputMDA = default;
        private StridedData<float3> inputVertices = default;
        private StridedData<float3> inputNormals = default;
        private NativeArray<Matrix4x4> inputBindposes = default;
        private NativeArray<BoneWeight1>.ReadOnly boneWeights = default;
        private NativeArray<byte> weightsPerVertex = default;
        private NativeArray<int> weightsPerVertexScan = default;
        private NativeArray<float> initialVertexDistances = default;
        private JobHandle vertexDistanceCalcHandle = default;
        private bool isScheduled = false;
        private JobHandle skinningHandle = default;
        private NativeArray<float4x4> currentBoneMatrices = default;
        private NativeHashSet<int> excludeVAttributesDuringCopy = default;
#if UNITY_EDITOR
        private Mesh previousSharedMesh = null;
        private Transform previousRootBone = null;
#endif

        #region API
        /// <summary>
        /// Original mesh in bindpose (T-Pose)
        /// </summary>
        public Mesh SharedMesh
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => sharedMesh;

            set
            {
                if (sharedMesh != value)
                {
                    sharedMesh = value;
                    SharedMeshChanged();
                    ResetBones();
                }
            }
        }


        /// <summary>
        /// Root bone of the skeleton/rig (equivalent of <see cref="SkinnedMeshRenderer.rootBone"/>)
        /// </summary>
        public Transform RootBone
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => rootBone;

            set
            {
                if (rootBone != value)
                {
                    rootBone = value;
                    RigChanged();
                    ResetBones();
                }
            }
        }


        /// <summary>
        /// Equivalent of <see cref="SkinnedMeshRenderer.bones"/>
        /// </summary>
        public IEnumerable<Transform> BoneTransforms => boneTransforms;


        /// <summary>
        /// Method/Algorithm used for skinning
        /// </summary>
        public SkinningMethod Method
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => method;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => method = value;
        }


        /// <summary>
        /// Amount of optimization applied after DQS to compensate for excessive bulging
        /// </summary>
        public float BulgeOptimizationFactor
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => bulgeOptimizationFactor;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => bulgeOptimizationFactor = Mathf.Clamp01(value);
        }


        /// <summary>
        /// Update vertices and mesh bounds even when the renderer is invisible?
        /// </summary>
        public bool AlwaysUpdate
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => alwaysUpdate;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => alwaysUpdate = value;
        }


        /// <summary>
        /// Skinned Mesh output
        /// </summary>
        public Mesh OutputMesh => outputMesh;


        /// <summary>
        /// Schedule a skinning operation (should be force completed within 4 frames in compliance with Unity's <see cref="Allocator.TempJob"/> allocator)
        /// </summary>
        public void ScheduleSkinning(JobHandle dependency = default)
        {
            if (isScheduled)
            {
#if UNITY_ASSERTIONS
                string message = $"Skinning was already scheduled! ({this})";
                if (isActiveAndEnabled) message += "\nThis component is enabled. Skinning was likely automatically scheduled during MonoBehaviour.Update()" +
                        "- disable this component to manage skinning manually.";
                Debug.LogWarning(message);
#endif // UNITY_ASSERTIONS

                return;
            }

#if UNITY_ASSERTIONS
            if (mf == null) { Debug.LogError($"A MeshFilter component is required on this GameObject! ({this})"); return; }
            if (mr == null) { Debug.LogError($"A MeshRenderer component is required on this GameObject! ({this})"); return; };
            if (outputMesh == null) { Debug.LogError($"The output mesh is missing! ({this})"); return; };
            if (rootBone == null || boneTransforms.Length <= 0) { Debug.LogError($"Root bone and/or skeleton is missing! ({this})"); return; };
            if (rootBone.parent == null) { Debug.LogError($"Root bone cannot be at the top of the scene hierarchy! ({this})"); return; };
#endif // UNITY_ASSERTIONS

            ScheduleSkinning_Internal(dependency);
        }


        /// <summary>
        /// Complete any scheduled skinning operation. This method does not throw errors when there is nothing to complete.
        /// </summary>
        public void CompleteSkinning()
        {
            if (!isScheduled)
                return;

            isScheduled = false;

            // complete job
            skinningHandle.Complete();
            skinningHandle = default;

            // set submesh
            Mesh.MeshData outputMD = outputMDA[0];
            outputMD.subMeshCount = 1;
            outputMD.SetSubMesh(0, new SubMeshDescriptor(0, outputIndexCount, MeshTopology.Triangles), NO_VALIDATION_UPDATE_FLAGS);

            // apply changes
            Mesh.ApplyAndDisposeWritableMeshData(outputMDA, outputMesh, NO_VALIDATION_UPDATE_FLAGS);
            outputMesh.bounds = outputBounds.Value;
            outputMesh.UploadMeshData(false);
            outputMesh.MarkModified();

            // cleanup
            excludeVAttributesDuringCopy.Dispose();
        }
        #endregion

        #region INTERNAL
        private void Awake()
        {
            if (!TryGetComponent(out mf))
                Debug.LogWarning($"A MeshFilter component is required on this GameObject! ({this})");

            if (!TryGetComponent(out mr))
                Debug.LogWarning($"A MeshRenderer component is required on this GameObject! ({this})");

            if (mf != null)
            {
                outputMesh = new Mesh();
                outputMesh.MarkDynamic();
                mf.sharedMesh = outputMesh;
            }
            
            outputBounds = new NativeReference<Bounds>(Allocator.Persistent, NativeArrayOptions.ClearMemory);

            previousSharedMesh = sharedMesh;
            previousRootBone = rootBone;
            SharedMeshChanged();
            RigChanged();
            ResetBones();
        }


        private void Update()
        {
#if UNITY_EDITOR
            // react to inspector-caused value/reference changes that need to trigger other events
            if (sharedMesh != previousSharedMesh || rootBone != previousRootBone)
            {
                SharedMeshChanged();
                RigChanged();
                ResetBones();
                previousSharedMesh = sharedMesh;
                previousRootBone = rootBone;
            }
#endif // UNITY_EDITOR

            if (alwaysUpdate || mr.isVisible)
            {
                ScheduleSkinning_Internal(default); // no job dependency
            }
        }


        private void LateUpdate()
        {
            CompleteSkinning();
        }


        private void OnDestroy()
        {
            // prevent memory leaks by making sure no async operations are running
            vertexDistanceCalcHandle.Complete();
            CompleteSkinning();

            if (outputBounds.IsCreated) outputBounds.Dispose();
            if (inputMDA.Length != 0) inputMDA.Dispose();
            if (bones.IsCreated) bones.Dispose();
            if (currentBoneMatrices.IsCreated) currentBoneMatrices.Dispose();
            if (weightsPerVertexScan.IsCreated) weightsPerVertexScan.Dispose();
            if (initialVertexDistances.IsCreated) initialVertexDistances.Dispose();
        }


        private void ScheduleSkinning_Internal(JobHandle dependency)
        {
            // last fail-safe in release builds
            if (isScheduled
             || mf == null
             || mr == null
             || outputMesh == null
             || boneTransforms.Length <= 0
             || rootBone == null)
                return;

            isScheduled = true;

            // complete setup
            vertexDistanceCalcHandle.Complete();
            vertexDistanceCalcHandle = default;

            // allocate writable mesh data
            outputMDA = Mesh.AllocateWritableMeshData(1);
            Mesh.MeshData outputMD = outputMDA[0];
            outputMD.SetVertexBufferParams(sharedMesh.vertexCount, outputVertexAttributes);
            outputMD.SetIndexBufferParams(outputIndexCount, sharedMesh.indexFormat);

            excludeVAttributesDuringCopy = new NativeHashSet<int>(2, Allocator.TempJob) { UnsafeUtility.EnumToInt(VertexAttribute.Position) };
            if (enableNormalSkinning) excludeVAttributesDuringCopy.Add(UnsafeUtility.EnumToInt(VertexAttribute.Normal));

            // update bone transforms
            for (int i = 0; i < currentBoneMatrices.Length; i++)
                currentBoneMatrices[i] = (float4x4)boneTransforms[i].localToWorldMatrix;

            // create and schedule jobs
            var jCopy = new CopyVertexDataJob()
            {
                source = inputMDA[0],
                target = outputMD,
                excludeAttributes = excludeVAttributesDuringCopy.AsReadOnly()
            };
            JobHandle jhCopy = jCopy.ScheduleByRef(dependency);
            skinningHandle = JobHandle.CombineDependencies(BurstSkinningUtility.Skin(this, dependency), jhCopy);
        }


        private void SharedMeshChanged()
        {
            if (inputMDA.Length > 0) inputMDA.Dispose();
            if (weightsPerVertexScan.IsCreated) weightsPerVertexScan.Dispose();
            outputVertexBufferStride = 0;

            if (sharedMesh != null && !sharedMesh.isReadable)
            {
                Debug.LogError($"Shared Mesh {sharedMesh} does not have read/write enabled!");
                sharedMesh = null;
                return;
            }

            // get input mesh snapshot
            if (sharedMesh != null)
            {
                outputVertexAttributes = sharedMesh.GetVertexAttributes()
                    .Where(a => a.attribute is not (VertexAttribute.BlendIndices or VertexAttribute.BlendWeight) && sharedMesh.GetVertexAttributeStream(a.attribute) == 0)
                    .ToArray();

                outputIndexCount = 0;
                for (int i = 0; i < sharedMesh.subMeshCount; i++)
                    outputIndexCount += (int)sharedMesh.GetIndexCount(i);

                enableNormalSkinning = outputVertexAttributes.Any(a => a.attribute == VertexAttribute.Normal);

                for (int i = 0; i < outputVertexAttributes.Length; i++)
                    outputVertexBufferStride += SizeofVertexAttributeFormat(outputVertexAttributes[i].format) * outputVertexAttributes[i].dimension;

                inputMDA = Mesh.AcquireReadOnlyMeshData(sharedMesh);
                Mesh.MeshData md = inputMDA[0];
                int stride = md.GetVertexBufferStride(0);
                byte* ptr_vdata = (byte*)md.GetVertexData<byte>().GetUnsafeReadOnlyPtr();
                inputVertices = new StridedData<float3>(ptr_vdata + md.GetVertexAttributeOffset(VertexAttribute.Position), stride);
                inputNormals = new StridedData<float3>(ptr_vdata + md.GetVertexAttributeOffset(VertexAttribute.Normal), stride);

                inputBindposes = sharedMesh.GetBindposes();
                boneWeights = sharedMesh.GetAllBoneWeights().AsReadOnly();
                weightsPerVertex = sharedMesh.GetBonesPerVertex();
                weightsPerVertexScan = new NativeArray<int>(weightsPerVertex.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

                // burst compiled scan
                new WPVScanJob()
                {
                    weightsPerVertex = weightsPerVertex,
                    scan = weightsPerVertexScan
                }
                .Run();
            }
            else
            {
                outputVertexAttributes = Array.Empty<VertexAttributeDescriptor>();
                outputIndexCount = 0;
                enableNormalSkinning = false;

                inputMDA = default;
                inputVertices = default;
                inputNormals = default;
                inputBindposes = default;
                boneWeights = default;
                weightsPerVertex = default;
                weightsPerVertexScan = default;
            }
        }


        private void RigChanged()
        {
            if (rootBone == null || rootBone.childCount <= 0)
            {
                boneTransforms = Array.Empty<Transform>();
                return;
            }

            boneTransforms = BurstSkinningUtility.ExtractRig(transform, rootBone, inputBindposes);
        }


        private void ResetBones()
        {
            if (bones.IsCreated) bones.Dispose();
            if (currentBoneMatrices.IsCreated) currentBoneMatrices.Dispose();
            if (initialVertexDistances.IsCreated) initialVertexDistances.Dispose();

            if (sharedMesh == null || rootBone == null || boneTransforms.Length != sharedMesh.bindposeCount)
                return;

            // create native bones
            bones = new NativeArray<Bone>(boneTransforms.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            currentBoneMatrices = new NativeArray<float4x4>(boneTransforms.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < boneTransforms.Length; i++)
            {
                Matrix4x4 tf = rootBone.worldToLocalMatrix * boneTransforms[i].localToWorldMatrix;
                bones[i] = new()
                {
                    worldToBindpose = inputBindposes[i],
                    bindposeToSkinned = float4x4.identity,
                    bindposeToSkinnedDQ = DualQuaternion.Identity,
                    boundA = (float3)tf.MultiplyPoint3x4(Vector3.zero),
                    boundB = (float3)tf.MultiplyPoint3x4(boneTransforms[i].localPosition),
                    transformedA = float3.zero,
                    transformedAB = float3.zero,
                    dotTransformedAB = 0f
                };
                currentBoneMatrices[i] = float4x4.identity;
            }

            // schedule calculation of initial vertex-to-bone distances
            initialVertexDistances = new NativeArray<float>(sharedMesh.vertexCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            var j1 = new ComputeVertexToBoneDistancesParallelJob()
            {
                bones = bones.AsReadOnly(),
                distances = initialVertexDistances,
                vertices = inputVertices,
                weights = boneWeights,
                weightsPerVertexScan = weightsPerVertexScan.AsReadOnly()
            };
            vertexDistanceCalcHandle = j1.ScheduleByRef(sharedMesh.vertexCount, 128);
        }


        private static int SizeofVertexAttributeFormat(VertexAttributeFormat format)
        {
            return format switch
            {
                VertexAttributeFormat.Float16 => sizeof(short),
                VertexAttributeFormat.Float32 => sizeof(float),
                VertexAttributeFormat.SInt8 => sizeof(sbyte),
                VertexAttributeFormat.SInt16 => sizeof(short),
                VertexAttributeFormat.SInt32 => sizeof(int),
                VertexAttributeFormat.SNorm8 => sizeof(sbyte),
                VertexAttributeFormat.SNorm16 => sizeof(short),
                VertexAttributeFormat.UInt8 => sizeof(byte),
                VertexAttributeFormat.UInt16 => sizeof(ushort),
                VertexAttributeFormat.UInt32 => sizeof(uint),
                VertexAttributeFormat.UNorm8 => sizeof(byte),
                VertexAttributeFormat.UNorm16 => sizeof(ushort),
                _ => 0
            };
        }
        #endregion INTERNAL

        #region JOBS
        [BurstCompile]
        private struct WPVScanJob : IJob
        {
            [ReadOnly] public NativeArray<byte> weightsPerVertex;
            [WriteOnly] public NativeArray<int> scan;

            public void Execute()
            {
                Debug.Assert(weightsPerVertex.Length == scan.Length);

                int sum = 0;
                for (int i = 0; i < weightsPerVertex.Length; i++)
                {
                    scan[i] = sum;
                    sum += weightsPerVertex[i];
                }
            }
        }


        [BurstCompile]
        private struct CopyVertexDataJob : IJob
        {
            [ReadOnly] public NativeHashSet<int>.ReadOnly excludeAttributes;
            [ReadOnly] public Mesh.MeshData source;
            public Mesh.MeshData target;


            // EXECUTE
            public void Execute()
            {
                int vertexCount = source.vertexCount;
                byte* ptr_source = (byte*)source.GetVertexData<byte>().GetUnsafeReadOnlyPtr();
                byte* ptr_target = (byte*)target.GetVertexData<byte>().GetUnsafePtr();
                int srcStride = source.GetVertexBufferStride(0);
                int destStride = target.GetVertexBufferStride(0);

                // copy vertex data
                for (int i = 0; i < UnsafeUtility.EnumToInt(VertexAttribute.BlendWeight); i++)
                {
                    VertexAttribute attr = UnsafeUtility.As<int, VertexAttribute>(ref i);

                    // ensure the attribute should not be excluded
                    if (excludeAttributes.Contains(i))
                        continue;

                    // ensure both meshes have the attribute
                    if (!(source.HasVertexAttribute(attr) && target.HasVertexAttribute(attr)))
                        continue;

                    int srcOffset = source.GetVertexAttributeOffset(attr);
                    int srcSize = SizeofVertexAttributeFormat(source.GetVertexAttributeFormat(attr)) * source.GetVertexAttributeDimension(attr);
                    int destOffset = target.GetVertexAttributeOffset(attr);
                    int destSize = SizeofVertexAttributeFormat(target.GetVertexAttributeFormat(attr)) * target.GetVertexAttributeDimension(attr);

                    if (Hint.Unlikely(srcSize != destSize))
                        throw new Exception($"Incompatible source/target vertex formats in {nameof(CopyVertexDataJob)}");

                    // exclusively copy single attribute
                    UnsafeUtility.MemCpyStride(ptr_target + destOffset, destStride, ptr_source + srcOffset, srcStride, srcSize, vertexCount);
                }

                // copy indices
                var srcIndexData = source.GetIndexData<byte>();
                UnsafeUtility.MemCpy(target.GetIndexData<byte>().GetUnsafePtr(), srcIndexData.GetUnsafeReadOnlyPtr(), srcIndexData.Length);
            }
        }
        #endregion JOBS

        #region INTERFACE IMPL
        SkinningMethod IBurstSkinnable.GetSkinningMethod() => method;
        float IBurstSkinnable.GetBulgeOptimizationFactor() => bulgeOptimizationFactor;
        bool IBurstSkinnable.GetEnableNormalSkinning() => enableNormalSkinning;
        NativeArray<Bone> IBurstSkinnable.GetBones() => bones;
        NativeArray<float4x4> IBurstSkinnable.GetBoneTransforms() => currentBoneMatrices;
        float4x4 IBurstSkinnable.GetWorldToRoot() => (float4x4)rootBone.parent.worldToLocalMatrix;
        NativeArray<BoneWeight1>.ReadOnly IBurstSkinnable.GetWeights() => boneWeights;
        NativeArray<byte>.ReadOnly IBurstSkinnable.GetWeightsPerVertex() => weightsPerVertex.AsReadOnly();
        NativeArray<int>.ReadOnly IBurstSkinnable.GetWeightsPerVertexScan() => weightsPerVertexScan.AsReadOnly();
        NativeArray<float>.ReadOnly IBurstSkinnable.GetVertexDistancesToBone() => initialVertexDistances.AsReadOnly();
        bool IBurstSkinnable.GetEnableBoundsCalc() => true;
        NativeReference<Bounds> IBurstSkinnable.GetBoundsRef() => outputBounds;
        int IBurstSkinnable.GetVertexCount() => sharedMesh != null ? sharedMesh.vertexCount : 0;
        StridedData<float3> IBurstSkinnable.GetInputVertices() => inputVertices;
        StridedData<float3> IBurstSkinnable.GetInputNormals() => enableNormalSkinning ? inputNormals : default;

        StridedData<float3> IBurstSkinnable.GetOutputVertices()
        {
            Mesh.MeshData md = outputMDA[0];
            return new((byte*)md.GetVertexData<byte>().GetUnsafePtr() + md.GetVertexAttributeOffset(VertexAttribute.Position), outputVertexBufferStride);
        }

        StridedData<float3> IBurstSkinnable.GetOutputNormals()
        {
            if (!enableNormalSkinning)
                return default;

            Mesh.MeshData md = outputMDA[0];
            return new StridedData<float3>((byte*)md.GetVertexData<byte>().GetUnsafePtr() + md.GetVertexAttributeOffset(VertexAttribute.Normal), outputVertexBufferStride);
        }
        #endregion INTERFACE IMPL

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            // TODO:
            // some sort of preview like for SkinnedMeshRenderer
        }
#endif // UNITY_EDITOR
    }
}
