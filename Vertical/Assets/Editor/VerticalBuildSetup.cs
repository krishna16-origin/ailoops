using UnityEditor;
using UnityEngine;

namespace Vertical.Editor
{
    public static class VerticalBuildSetup
    {
        [MenuItem("Tools/Vertical/Configure Android Player")]
        public static void ConfigureAndroidPlayer()
        {
            PlayerSettings.productName = "Vertical";
            PlayerSettings.companyName = "Krishna";
            PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, "com.krishna.vertical");
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel24;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARMv7 | AndroidArchitecture.ARM64;
            PlayerSettings.Android.enableSustainedPerformanceMode = true;
            PlayerSettings.Android.blitType = AndroidBlitType.Always;
            EditorUserBuildSettings.buildAppBundle = false;
            AssetDatabase.SaveAssets();
            Debug.Log("Vertical Android player settings configured. Set a release keystore, then build an APK in File > Build Settings.");
        }

        [MenuItem("Tools/Vertical/Validate Project Files")]
        public static void ValidateProjectFiles()
        {
            var scene = AssetDatabase.LoadAssetAtPath<SceneAsset>("Assets/Scenes/Vertical.unity");
            if (scene == null)
            {
                Debug.LogError("Vertical scene was not found. Open Assets/Scenes/Vertical.unity.");
                return;
            }

            if (Shader.Find("Universal Render Pipeline/Lit") == null && Shader.Find("Standard") == null)
            {
                Debug.LogError("No supported lit shader was found.");
                return;
            }

            Debug.Log("Vertical source validation passed. Enter Play mode to verify the gameplay route.");
        }
    }
}
