using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace TriceHelix.BurstSkinning.Core
{
    [BurstCompile]
    public struct ComputeBoundsJob : IJob
    {
        // input
        public int vertexCount;
        public StridedData<float3> vertices;

        // output
        [WriteOnly] public NativeReference<Bounds> ref_bounds;


        public void Execute()
        {
            float3 min = float.MaxValue;
            float3 max = float.MinValue;

            for (int i = 0; i < vertexCount; i++)
            {
                float3 v = vertices[i];
                min = math.min(min, v);
                max = math.max(max, v);
            }

            ref_bounds.Value = (vertexCount > 0) ? new Bounds(math.lerp(min, max, 0.5f), max - min) : default;
        }
    }
}
