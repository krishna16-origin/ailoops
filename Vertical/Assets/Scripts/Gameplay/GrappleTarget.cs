using System.Collections.Generic;
using UnityEngine;

namespace Vertical
{
    public sealed class GrappleTarget : MonoBehaviour
    {
        public static readonly List<GrappleTarget> ActiveTargets = new List<GrappleTarget>(24);

        private Renderer cachedRenderer;
        private Vector3 baseScale;

        private void Awake()
        {
            cachedRenderer = GetComponent<Renderer>();
            baseScale = transform.localScale;
            ActiveTargets.Add(this);
        }

        private void OnDestroy() => ActiveTargets.Remove(this);

        private void Update()
        {
            var pulse = 1f + Mathf.Sin(Time.time * 3.8f + transform.position.y) * 0.08f;
            transform.localScale = baseScale * pulse;
            transform.Rotate(0f, 54f * Time.deltaTime, 0f, Space.Self);
        }

        public void SetHighlighted(bool highlighted)
        {
            if (cachedRenderer == null)
                return;

            cachedRenderer.material.SetColor("_EmissionColor", highlighted ? VerticalBootstrap.Cyan * 7.5f : VerticalBootstrap.Cyan * 3.2f);
        }
    }
}
