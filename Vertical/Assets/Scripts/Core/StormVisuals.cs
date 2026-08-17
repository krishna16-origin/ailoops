using UnityEngine;

namespace Vertical
{
    public static class StormVisuals
    {
        public static void BuildRain(Transform followTarget)
        {
            var rainObject = new GameObject("Storm Rain");
            var rain = rainObject.AddComponent<ParticleSystem>();
            var main = rain.main;
            main.loop = true;
            main.playOnAwake = true;
            main.maxParticles = 500;
            main.startLifetime = 1.1f;
            main.startSpeed = 25f;
            main.startSize = 0.045f;
            main.startColor = new Color(0.65f, 0.85f, 0.92f, 0.45f);
            main.simulationSpace = ParticleSystemSimulationSpace.World;

            var emission = rain.emission;
            emission.rateOverTime = 360f;
            var shape = rain.shape;
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = new Vector3(28f, 1f, 20f);
            var velocity = rain.velocityOverLifetime;
            velocity.enabled = true;
            velocity.y = -14f;
            velocity.x = 1.4f;
            var renderer = rain.GetComponent<ParticleSystemRenderer>();
            renderer.renderMode = ParticleSystemRenderMode.Stretch;
            renderer.lengthScale = 3.8f;
            renderer.velocityScale = 0.5f;

            rainObject.AddComponent<FollowRain>().Configure(followTarget);
        }
    }

    public sealed class FollowRain : MonoBehaviour
    {
        private Transform target;
        public void Configure(Transform targetTransform) => target = targetTransform;

        private void LateUpdate()
        {
            if (target != null)
                transform.position = target.position + new Vector3(0f, 12f, 0f);
        }
    }
}
