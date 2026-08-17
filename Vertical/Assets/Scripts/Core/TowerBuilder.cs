using UnityEngine;

namespace Vertical
{
    public static class TowerBuilder
    {
        private const int Floors = 18;
        private const float FloorHeight = 5f;

        public static void Build(MissionController mission)
        {
            var facadeMaterial = MaterialFactory.CreateFacadeMaterial();
            var trimMaterial = MaterialFactory.CreateEmissive(VerticalBootstrap.Cyan);
            var warningMaterial = MaterialFactory.CreateEmissive(VerticalBootstrap.Amber);

            CreateCube("Tower Core", new Vector3(0f, 43f, 2.5f), new Vector3(32f, 92f, 4f), facadeMaterial, true);

            for (var floor = 0; floor < Floors; floor++)
            {
                var y = floor * FloorHeight;
                CreateCube("Facade Beam", new Vector3(0f, y + 1.2f, -0.1f), new Vector3(32f, 0.35f, 0.6f), facadeMaterial, true);

                if (floor % 2 == 0)
                {
                    CreateCube("Cyan Signal", new Vector3(-13.8f, y + 2.8f, -0.35f), new Vector3(0.32f, 1.4f, 0.18f), trimMaterial, false);
                    CreateCube("Cyan Signal", new Vector3(13.8f, y + 2.8f, -0.35f), new Vector3(0.32f, 1.4f, 0.18f), trimMaterial, false);
                }

                if (floor == 0 || floor == 8 || floor == 15)
                    CreateLedge(new Vector3(0f, y + 0.35f, -3.2f), warningMaterial);
            }

            var anchors = new[]
            {
                new Vector3(-6.5f, 7f, -2.8f), new Vector3(5.5f, 13f, -3.1f), new Vector3(-7.5f, 20f, -2.8f),
                new Vector3(6.0f, 28f, -3.1f), new Vector3(-5.5f, 37f, -2.8f), new Vector3(7.0f, 46f, -3.1f),
                new Vector3(-7.0f, 56f, -2.8f), new Vector3(4.5f, 66f, -3.1f), new Vector3(-3.0f, 75f, -2.8f)
            };

            for (var i = 0; i < anchors.Length; i++)
                CreateGrappleTarget(anchors[i], trimMaterial, 1.2f + i * 0.12f);

            var objective = CreateCube("Objective Ledge", new Vector3(-3f, 78.2f, -4.2f), new Vector3(9f, 0.65f, 4f), warningMaterial, true);
            objective.AddComponent<ObjectiveLedge>().Configure(mission);

            var moon = new GameObject("Moon Key Light");
            moon.transform.rotation = Quaternion.Euler(38f, -32f, 0f);
            var light = moon.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = new Color(0.35f, 0.67f, 0.90f);
            light.intensity = 0.85f;
            light.shadows = LightShadows.Soft;
            light.shadowStrength = 0.55f;

            for (var i = 0; i < 11; i++)
            {
                var mist = CreateCube("Fog Layer", new Vector3((i % 2 == 0 ? -1f : 1f) * 10f, 14f + i * 8.5f, 7f), new Vector3(26f, 2.5f, 0.15f), MaterialFactory.CreateEmissive(new Color(0.02f, 0.13f, 0.17f)), false);
                mist.GetComponent<Collider>().enabled = false;
            }
        }

        private static GameObject CreateCube(string objectName, Vector3 position, Vector3 scale, Material material, bool solid)
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = objectName;
            cube.transform.position = position;
            cube.transform.localScale = scale;
            cube.GetComponent<Renderer>().sharedMaterial = material;
            if (!solid)
                cube.GetComponent<Collider>().enabled = false;
            return cube;
        }

        private static void CreateLedge(Vector3 position, Material material)
        {
            var ledge = CreateCube("Checkpoint Ledge", position, new Vector3(15f, 0.55f, 4.3f), material, true);
            ledge.AddComponent<CheckpointLedge>();
        }

        private static void CreateGrappleTarget(Vector3 position, Material material, float scale)
        {
            var target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            target.name = "Grapple Target";
            target.transform.position = position;
            target.transform.localScale = Vector3.one * scale;
            target.GetComponent<Renderer>().sharedMaterial = material;
            target.GetComponent<SphereCollider>().radius = 0.75f;
            target.AddComponent<GrappleTarget>();
        }
    }
}
