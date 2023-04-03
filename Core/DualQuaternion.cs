using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace TriceHelix.BurstSkinning.Core
{
    [Serializable]
    public struct DualQuaternion : IEquatable<DualQuaternion>
    {
        public static readonly DualQuaternion Identity = new(quaternion.identity, new quaternion(0f, 0f, 0f, 0f));

        public quaternion real;
        public quaternion dual;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public DualQuaternion(quaternion real, quaternion dual)
        {
            this.real = real;
            this.dual = dual;
        }


        public DualQuaternion(quaternion real, float3 translation)
        {
            this.real = real;
            dual = 0.5f * math.mul(new quaternion(translation.x, translation.y, translation.z, 0f), real).value;
        }


        // INTERFACE
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(DualQuaternion other)
        {
            return real.Equals(other.real) && dual.Equals(other.dual);
        }


        // OVERRIDES
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override bool Equals(object obj)
        {
            return Equals((DualQuaternion)obj);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override int GetHashCode()
        {
            return System.HashCode.Combine(real, dual);
        }


        // OPERATORS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(DualQuaternion lhs, DualQuaternion rhs)
        {
            return Equals(lhs, rhs);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(DualQuaternion lhs, DualQuaternion rhs)
        {
            return !Equals(lhs, rhs);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DualQuaternion operator +(DualQuaternion lhs, DualQuaternion rhs)
        {
            lhs.real.value += rhs.real.value;
            lhs.dual.value += rhs.dual.value;
            return lhs;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DualQuaternion operator -(DualQuaternion lhs, DualQuaternion rhs)
        {
            lhs.real.value -= rhs.real.value;
            lhs.dual.value -= rhs.dual.value;
            return lhs;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DualQuaternion operator *(DualQuaternion lhs, float rhs)
        {
            lhs.real.value *= rhs;
            lhs.dual.value *= rhs;
            return lhs;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DualQuaternion operator *(float lhs, DualQuaternion rhs)
        {
            rhs.real.value *= lhs;
            rhs.dual.value *= lhs;
            return rhs;
        }


        public static DualQuaternion operator *(DualQuaternion lhs, DualQuaternion rhs)
        {
            quaternion tmp = lhs.real;
            lhs.real = math.mul(tmp, rhs.real);
            lhs.dual.value = math.mul(tmp, rhs.dual).value + math.mul(lhs.dual, rhs.real).value;
            return lhs;
        }


        // METHODS
        public DualQuaternion Inverse()
        {
            DualQuaternion inv = default;
            inv.real = math.inverse(real);
            inv.dual.value = -math.mul(inv.real, math.mul(dual, inv.real)).value;
            return inv;
        }


        public DualQuaternion Normalized()
        {
            DualQuaternion normal = default;
            float4 d = dual.value / math.length(real);
            normal.real = math.normalize(real);
            normal.dual = d - (math.dot(normal.real.value, d) * normal.real.value);
            return normal;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public DualQuaternion Difference(DualQuaternion dq)
        {
            return Inverse() * dq;
        }


        public float3 Transform(float3 point)
        {
            DualQuaternion a = this * new DualQuaternion(quaternion.identity, new quaternion(point.x, point.y, point.z, 0f));
            DualQuaternion b = new(math.conjugate(real), new quaternion(-math.conjugate(dual).value));
            return (a * b).dual.value.xyz;
        }
    }
}
