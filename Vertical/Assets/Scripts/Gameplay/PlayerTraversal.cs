using System;
using UnityEngine;

namespace Vertical
{
    public enum TraversalState { Grounded, Airborne, Swinging, Gliding, Complete }

    [RequireComponent(typeof(Rigidbody))]
    public sealed class PlayerTraversal : MonoBehaviour
    {
        [Header("Mobile tuning")]
        [SerializeField] private float maxGrappleDistance = 30f;
        [SerializeField] private float gravity = 22f;
        [SerializeField] private float swingAcceleration = 17f;
        [SerializeField] private float launchBoost = 3.5f;
        [SerializeField] private float glideLift = 13.5f;
        [SerializeField] private float glideDrag = 0.75f;
        [SerializeField] private float targetForgivenessPixels = 220f;

        public event Action<float> AltitudeChanged;
        public event Action<TraversalState> StateChanged;

        public TraversalState State { get; private set; } = TraversalState.Grounded;
        public float Altitude => Mathf.Max(0f, transform.position.y - startPoint.y);
        public bool IsSwinging => State == TraversalState.Swinging;
        public bool CanAcceptGameplayInput => State != TraversalState.Complete;

        private Rigidbody body;
        private Transform cameraTransform;
        private LineRenderer rope;
        private GrappleTarget attachedTarget;
        private GrappleTarget highlightedTarget;
        private float ropeLength;
        private float steering;
        private bool glideRequested;
        private Vector3 startPoint;
        private Vector3 checkpoint;
        private float lastAltitude;

        public void Configure(Transform activeCamera)
        {
            body = GetComponent<Rigidbody>();
            cameraTransform = activeCamera;
            startPoint = transform.position;
            checkpoint = startPoint;

            var cameraFollow = cameraTransform.GetComponent<CameraFollow>();
            cameraFollow.Configure(transform);

            rope = gameObject.AddComponent<LineRenderer>();
            rope.positionCount = 2;
            rope.startWidth = 0.075f;
            rope.endWidth = 0.035f;
            rope.useWorldSpace = true;
            rope.sharedMaterial = new Material(Shader.Find("Sprites/Default"));
            rope.sharedMaterial.color = VerticalBootstrap.Cyan;
            rope.enabled = false;
        }

        private void Update()
        {
            if (Input.GetKey(KeyCode.A)) steering = -1f;
            else if (Input.GetKey(KeyCode.D)) steering = 1f;
            else if (!Input.GetMouseButton(0)) steering = Mathf.MoveTowards(steering, 0f, Time.deltaTime * 5f);

            if (Input.GetKeyDown(KeyCode.Space))
                TryGrapple(new Vector2(Screen.width * 0.5f, Screen.height * 0.65f));
            if (Input.GetKeyDown(KeyCode.R))
                RespawnAtCheckpoint();

            if (rope.enabled && attachedTarget != null)
            {
                rope.SetPosition(0, transform.position + Vector3.up * 0.5f);
                rope.SetPosition(1, attachedTarget.transform.position);
            }

            if (transform.position.y < checkpoint.y - 10f)
                RespawnAtCheckpoint();

            var altitude = Altitude;
            if (Mathf.Abs(altitude - lastAltitude) > 0.05f)
            {
                lastAltitude = altitude;
                AltitudeChanged?.Invoke(altitude);
            }
        }

        private void FixedUpdate()
        {
            if (State == TraversalState.Complete)
                return;

            if (State == TraversalState.Swinging && attachedTarget != null)
                SimulateSwing();
            else
                SimulateAir();

            OrientToVelocity();
        }

        private void SimulateAir()
        {
            if (State == TraversalState.Grounded)
            {
                body.velocity = Vector3.Lerp(body.velocity, Vector3.zero, Time.fixedDeltaTime * 2.5f);
                return;
            }

            var downwardAcceleration = State == TraversalState.Gliding ? gravity * 0.25f : gravity;
            body.AddForce(Vector3.down * downwardAcceleration, ForceMode.Acceleration);

            if (State == TraversalState.Gliding && glideRequested)
            {
                body.AddForce(Vector3.up * glideLift, ForceMode.Acceleration);
                body.AddForce(-body.velocity.normalized * glideDrag, ForceMode.Acceleration);
                var forward = Vector3.ProjectOnPlane(cameraTransform.forward, Vector3.up).normalized;
                body.AddForce(forward * 5.5f, ForceMode.Acceleration);
            }
        }

        private void SimulateSwing()
        {
            var pivot = attachedTarget.transform.position;
            var offset = transform.position - pivot;
            var distance = offset.magnitude;
            if (distance < 0.01f)
                return;

            var radial = offset / distance;
            var velocity = body.velocity;
            var radialSpeed = Vector3.Dot(velocity, radial);
            if (radialSpeed > 0f)
                velocity -= radial * radialSpeed;

            var desiredDirection = Vector3.ProjectOnPlane(cameraTransform.right * steering, radial).normalized;
            velocity += desiredDirection * swingAcceleration * Time.fixedDeltaTime;
            velocity += Vector3.down * gravity * Time.fixedDeltaTime;

            var excess = distance - ropeLength;
            if (excess > 0f)
                velocity -= radial * (excess / Time.fixedDeltaTime);

            body.velocity = Vector3.ClampMagnitude(velocity, 31f);
        }

        public void UpdateAim(Vector2 screenPoint, GameUI ui)
        {
            if (!CanAcceptGameplayInput || State == TraversalState.Swinging)
                return;

            var best = FindBestTarget(screenPoint);
            if (highlightedTarget == best)
            {
                ui.UpdateReticle(screenPoint, best != null);
                return;
            }

            if (highlightedTarget != null)
                highlightedTarget.SetHighlighted(false);
            highlightedTarget = best;
            if (highlightedTarget != null)
                highlightedTarget.SetHighlighted(true);
            ui.UpdateReticle(screenPoint, highlightedTarget != null);
        }

        public bool TryGrapple(Vector2 screenPoint)
        {
            if (!CanAcceptGameplayInput)
                return false;

            var target = FindBestTarget(screenPoint);
            if (target == null)
                return false;

            AttachTo(target);
            return true;
        }

        public void ReleaseGrapple(Vector2 launchDrag)
        {
            if (State != TraversalState.Swinging)
                return;

            var dragBias = new Vector3(launchDrag.x, launchDrag.y * 0.35f, 0f) * 0.025f;
            var launch = body.velocity + dragBias + Vector3.up * launchBoost;
            attachedTarget = null;
            rope.enabled = false;
            body.velocity = Vector3.ClampMagnitude(launch, 35f);
            SetState(TraversalState.Airborne);
        }

        public void SetGlide(bool requested)
        {
            glideRequested = requested;
            if (State == TraversalState.Airborne && requested && body.velocity.y <= 3f)
                SetState(TraversalState.Gliding);
            else if (State == TraversalState.Gliding && !requested)
                SetState(TraversalState.Airborne);
        }

        public void EmergencyRecover(Vector2 screenPoint)
        {
            if (State == TraversalState.Swinging)
            {
                ReleaseGrapple(Vector2.zero);
                return;
            }

            TryGrapple(screenPoint);
        }

        public void SetSteering(float value) => steering = Mathf.Clamp(value, -1f, 1f);

        public void SetCheckpoint(Vector3 checkpointPosition)
        {
            checkpoint = checkpointPosition + Vector3.up * 2.4f + Vector3.back * 1.8f;
        }

        public void RespawnAtCheckpoint()
        {
            attachedTarget = null;
            rope.enabled = false;
            body.velocity = Vector3.zero;
            transform.position = checkpoint;
            SetState(checkpoint == startPoint ? TraversalState.Grounded : TraversalState.Airborne);
        }

        public void Complete()
        {
            attachedTarget = null;
            rope.enabled = false;
            body.velocity = Vector3.zero;
            SetState(TraversalState.Complete);
        }

        private void AttachTo(GrappleTarget target)
        {
            attachedTarget = target;
            ropeLength = Vector3.Distance(transform.position, target.transform.position);
            rope.enabled = true;
            glideRequested = false;
            SetState(TraversalState.Swinging);
        }

        private GrappleTarget FindBestTarget(Vector2 screenPoint)
        {
            var camera = Camera.main;
            if (camera == null)
                return null;

            GrappleTarget best = null;
            var bestScore = float.MaxValue;
            for (var i = 0; i < GrappleTarget.ActiveTargets.Count; i++)
            {
                var candidate = GrappleTarget.ActiveTargets[i];
                if (candidate == null)
                    continue;

                var worldOffset = candidate.transform.position - transform.position;
                var distance = worldOffset.magnitude;
                if (distance > maxGrappleDistance || candidate.transform.position.y < transform.position.y - 2f)
                    continue;

                var viewportPoint = camera.WorldToScreenPoint(candidate.transform.position);
                if (viewportPoint.z <= 0f)
                    continue;

                var screenDistance = Vector2.Distance(screenPoint, new Vector2(viewportPoint.x, viewportPoint.y));
                if (screenDistance > targetForgivenessPixels)
                    continue;

                var score = screenDistance + distance * 1.35f - Mathf.Max(0f, worldOffset.y) * 3.2f;
                if (score < bestScore)
                {
                    best = candidate;
                    bestScore = score;
                }
            }
            return best;
        }

        private void SetState(TraversalState next)
        {
            if (State == next)
                return;

            State = next;
            StateChanged?.Invoke(State);
        }

        private void OrientToVelocity()
        {
            var flatVelocity = Vector3.ProjectOnPlane(body.velocity, Vector3.up);
            if (flatVelocity.sqrMagnitude < 0.4f)
                return;

            var targetRotation = Quaternion.LookRotation(flatVelocity.normalized, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.fixedDeltaTime * 6f);
        }
    }
}
