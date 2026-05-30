<?php

namespace App\Http\Controllers\Webhook;

use App\Http\Controllers\Controller;
use App\Models\Camera;
use App\Models\Notification;
use App\Models\Section;
use App\Models\SurveillanceEvent;
use App\Models\TeacherAssignment;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class AttentionAlertWebhookController extends Controller
{
    /**
     * POST /api/webhooks/attention-alert
     *
     * Receives a low-attention alert from the student monitoring AI service.
     * Authenticated by X-API-Key header matched against AI_API_KEY in .env.
     *
     * Expected payload:
     * {
     *   "type":                 "low_attention",
     *   "camera_id":            "classroom-cam-01",   // matches camera.code
     *   "low_attention_ratio":  0.73,                 // 0.0 – 1.0
     *   "student_count":        11,
     *   "duration_seconds":     120,
     *   "timestamp":            "2026-05-30T10:23:45Z",
     *   "section_id":           3                     // nullable
     * }
     */
    public function handle(Request $request)
    {
        if (!$this->validKey($request)) {
            return response()->json(['error' => 'Unauthorized'], 401);
        }

        $data = $request->validate([
            'type'                 => 'required|string',
            'camera_id'            => 'required|string',
            'low_attention_ratio'  => 'required|numeric|between:0,1',
            'student_count'        => 'required|integer|min:0',
            'duration_seconds'     => 'required|integer|min:0',
            'timestamp'            => 'required|date',
            'section_id'           => 'nullable|integer|exists:section,section_id',
            'student_id'           => 'nullable|integer|exists:students,id',
            'footage_path'         => 'nullable|string|max:512',
        ]);

        $ratio    = (float) $data['low_attention_ratio'];
        $camera   = Camera::where('code', $data['camera_id'])->first();
        $section  = $data['section_id'] ? Section::find($data['section_id']) : null;

        $event = DB::transaction(function () use ($data, $ratio, $camera, $section) {
            $event = SurveillanceEvent::create([
                'camera_id'          => $camera?->camera_id,
                'detectedtype'       => 'low_attention',
                'detectedat'         => Carbon::parse($data['timestamp']),
                'severity'           => $this->severityFromRatio($ratio),
                'confidence'         => $ratio,
                'relatedsection_id'  => $section?->section_id,
                'relatedstudent_id'  => $data['student_id'] ?? null,
                'footage_path'       => $data['footage_path'] ?? null,
                'status'             => 'new',
            ]);

            $this->notifyTeachers($camera, $data['camera_id'], $ratio, $data['student_count'], $section);

            return $event;
        });

        return response()->json([
            'message'  => 'Attention alert recorded.',
            'event_id' => $event->survevent_id,
        ], 201);
    }

    private function validKey(Request $request): bool
    {
        $expected = config('services.ai_api_key');
        $provided = $request->header('X-AI-Key', '');

        return $expected && hash_equals($expected, $provided);
    }

    private function severityFromRatio(float $ratio): string
    {
        return match (true) {
            $ratio >= 0.85 => 'critical',
            $ratio >= 0.70 => 'high',
            $ratio >= 0.55 => 'medium',
            default        => 'low',
        };
    }

    private function notifyTeachers(?Camera $camera, string $cameraCode, float $ratio, int $studentCount, ?Section $section): void
    {
        // Collect teacher user IDs: teachers assigned to this section (if known)
        $teacherUserIds = collect();

        if ($section) {
            $teacherUserIds = TeacherAssignment::where('section_id', $section->section_id)
                ->with('teacher.user')
                ->get()
                ->pluck('teacher.user.id')
                ->filter();
        }

        if ($teacherUserIds->isEmpty()) {
            return;
        }

        $location    = $camera ? $camera->location : $cameraCode;
        $pct         = (int) round($ratio * 100);
        $studentName = request()->input('student_name');

        $title = $studentName
            ? "Low Attention — {$studentName} | {$location}"
            : "Low Attention Alert — {$location}";

        $body = $studentName
            ? sprintf('%s is not paying attention (%d%%) | Camera: %s', $studentName, $pct, $cameraCode)
            : sprintf('%d%% of students show low attention | %d detected | Camera: %s', $pct, $studentCount, $cameraCode);

        $notification = Notification::create([
            'title'           => $title,
            'body'            => $body,
            'createdbyuserid' => null,
            'channel'         => 'alert',
        ]);

        DB::table('notificationrecipient')->insert(
            $teacherUserIds->map(fn ($uid) => [
                'notification_id' => $notification->notification_id,
                'user_id'         => $uid,
                'status'          => 'unread',
                'deliveredat'     => now(),
                'readat'          => null,
            ])->toArray()
        );
    }
}
