using UnityEngine;

namespace Vertical
{
    [DefaultExecutionOrder(-100)]
    public sealed class VerticalBootstrap : MonoBehaviour
    {
        public static readonly Color Void = new Color(0.027f, 0.039f, 0.059f);
        public static readonly Color Concrete = new Color(0.075f, 0.105f, 0.132f);
        public static readonly Color Cyan = new Color(0.30f, 0.90f, 1.00f);
        public static readonly Color Amber = new Color(1.00f, 0.71f, 0.34f);

        private PlayerTraversal player;
        private GameUI ui;
        private MissionController mission;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void CreateRuntime()
        {
            if (FindObjectOfType<VerticalBootstrap>() != null)
                return;

            var root = new GameObject("Vertical Runtime");
            root.AddComponent<VerticalBootstrap>();
        }

        private void Awake()
        {
            Application.targetFrameRate = 30;
            Screen.orientation = ScreenOrientation.Portrait;
            QualityController.ApplyStoredProfile();
            ConfigureEnvironment();

            var camera = CreateCamera();
            player = CreatePlayer(camera.transform);
            mission = gameObject.AddComponent<MissionController>();
            mission.Configure(player);
            gameObject.AddComponent<AudioSource>();
            gameObject.AddComponent<MobileFeedback>().Configure(player);
            TowerBuilder.Build(mission);
            StormVisuals.BuildRain(player.transform);

            ui = GameUI.Create(player, mission);
            mission.SetUI(ui);
            gameObject.AddComponent<TouchInput>().Configure(player, ui);
            ui.ShowTitle();
        }

        private static void ConfigureEnvironment()
        {
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = new Color(0.035f, 0.070f, 0.090f);
            RenderSettings.fogDensity = 0.018f;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.05f, 0.08f, 0.10f);
            RenderSettings.reflectionIntensity = 0.15f;
        }

        private static Camera CreateCamera()
        {
            var cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Void;
            camera.fieldOfView = 60f;
            camera.nearClipPlane = 0.15f;
            camera.farClipPlane = 160f;
            cameraObject.AddComponent<AudioListener>();
            cameraObject.AddComponent<CameraFollow>();
            return camera;
        }

        private static PlayerTraversal CreatePlayer(Transform cameraTransform)
        {
            var playerObject = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            playerObject.name = "Runner";
            playerObject.transform.position = new Vector3(0f, 2.4f, -7.5f);
            playerObject.transform.localScale = new Vector3(0.9f, 1.35f, 0.9f);
            playerObject.GetComponent<Renderer>().sharedMaterial = MaterialFactory.CreateBodyMaterial();

            var body = playerObject.AddComponent<Rigidbody>();
            body.useGravity = false;
            body.interpolation = RigidbodyInterpolation.Interpolate;
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            body.constraints = RigidbodyConstraints.FreezeRotation;

            var traversal = playerObject.AddComponent<PlayerTraversal>();
            traversal.Configure(cameraTransform);
            return traversal;
        }
    }

    public static class MaterialFactory
    {
        private static Shader LitShader => Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");

        public static Material CreateBodyMaterial()
        {
            var material = new Material(LitShader);
            material.color = new Color(0.04f, 0.06f, 0.075f);
            material.SetColor("_BaseColor", material.color);
            material.SetFloat("_Smoothness", 0.55f);
            return material;
        }

        public static Material CreateFacadeMaterial()
        {
            var material = new Material(LitShader);
            material.color = VerticalBootstrap.Concrete;
            material.SetColor("_BaseColor", material.color);
            material.SetFloat("_Metallic", 0.18f);
            material.SetFloat("_Smoothness", 0.48f);
            return material;
        }

        public static Material CreateEmissive(Color color)
        {
            var material = new Material(LitShader);
            material.EnableKeyword("_EMISSION");
            material.SetColor("_BaseColor", color * 0.2f);
            material.SetColor("_EmissionColor", color * 3.2f);
            material.SetFloat("_Smoothness", 0.7f);
            return material;
        }
    }
}
