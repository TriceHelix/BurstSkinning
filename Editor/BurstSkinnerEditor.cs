using UnityEngine;
using UnityEditor;
using TriceHelix.BurstSkinning.Core;

namespace TriceHelix.BurstSkinning.Editor
{
    [CustomEditor(typeof(BurstSkinner)), CanEditMultipleObjects]
    public class BurstSkinnerEditor : UnityEditor.Editor
    {
        private SerializedProperty p_sharedMesh = null;
        private SerializedProperty p_rootBone = null;
        private SerializedProperty p_method = null;
        private SerializedProperty p_bulgeOptimizationFactor = null;
        private SerializedProperty p_alwaysUpdate = null;


        private void OnEnable()
        {
            p_sharedMesh = serializedObject.FindProperty("sharedMesh");
            p_rootBone = serializedObject.FindProperty("rootBone");
            p_method = serializedObject.FindProperty("method");
            p_bulgeOptimizationFactor = serializedObject.FindProperty("bulgeOptimizationFactor");
            p_alwaysUpdate = serializedObject.FindProperty("alwaysUpdate");
        }


        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.PropertyField(p_sharedMesh);
            EditorGUILayout.PropertyField(p_rootBone);
            EditorGUILayout.PropertyField(p_alwaysUpdate);
            EditorGUILayout.PropertyField(p_method);

            // this property only applies to DQS skinning
            if (!p_bulgeOptimizationFactor.hasMultipleDifferentValues && p_method.enumValueIndex == (int)SkinningMethod.DQS)
                EditorGUILayout.PropertyField(p_bulgeOptimizationFactor);

            serializedObject.ApplyModifiedProperties();
        }
    }
}
