using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;

namespace TriceHelix.BurstSkinning.Core
{
    public unsafe readonly struct StridedData<T> where T : unmanaged
    {
        [NativeDisableUnsafePtrRestriction]
        public readonly void* Pointer;
        public readonly int Stride;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public StridedData(void* ptr, int stride)
        {
            Pointer = ptr;
            Stride = stride;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public StridedData(void* ptr) : this(ptr, sizeof(T)) { }


        public ref T this[int index] => ref UnsafeUtility.AsRef<T>((byte*)Pointer + (Stride * index));
    }
}
