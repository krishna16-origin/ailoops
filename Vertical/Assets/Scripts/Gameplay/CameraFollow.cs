using UnityEngine;

namespace Vertical
{
    public sealed class CameraFollow : MonoBehaviour
    {
        private Transform target;
        private Vector3 velocity;

        public void Configure(Transform followTarget) => target = followTarget;

        private void LateUpdate()
        {
            if (target == null)
                return;

            var player = target.GetComponent<PlayerTraversal>();
            var glide = player != null && player.State == TraversalState.Gliding;
            var desired = target.position + new Vector3(0f, glide ? 6.8f : 5.3f, glide ? -17.5f : -14.2f);
            transform.position = Vector3.SmoothDamp(transform.position, desired, ref velocity, 0.24f);
            var lookAt = target.position + Vector3.up * (glide ? 2.1f : 1.4f) + Vector3.forward * 4.5f;
            transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(lookAt - transform.position), Time.deltaTime * 7.5f);
            GetComponent<Camera>().fieldOfView = Mathf.Lerp(GetComponent<Camera>().fieldOfView, glide ? 68f : 60f, Time.deltaTime * 3f);
        }
    }
}
