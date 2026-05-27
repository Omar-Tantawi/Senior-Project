<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\AssessmentResult;
use App\Models\BehaviorLog;
use App\Models\ClassAssignment;
use App\Models\ClassDistribution;
use App\Models\Enrollment;
use App\Models\Section;
use App\Models\Student;
use App\Models\StudentAttendance;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Str;

class ClassDistributionController extends Controller
{
    public function distribute(Request $request)
    {
        $validated = $request->validate([
            'class_id'                    => 'required|integer',
            'num_classes'                 => 'required|integer|min:2|max:10',
            'criteria_weights'            => 'required|array',
            'criteria_weights.academic'   => 'required|numeric|min:0|max:1',
            'criteria_weights.behavior'   => 'required|numeric|min:0|max:1',
            'criteria_weights.attendance' => 'required|numeric|min:0|max:1',
        ]);

        $sectionIds = Section::where('class_id', $validated['class_id'])
            ->pluck('section_id');

        $studentIds = Enrollment::whereIn('section_id', $sectionIds)
            ->where('status', 'active')
            ->pluck('student_id')
            ->unique()
            ->values();

        if ($studentIds->isEmpty()) {
            return response()->json(['message' => 'No active students found in this class.'], 422);
        }

        $students = $studentIds->map(fn ($sid) => [
            'student_id'     => (string) $sid,
            'academic_score' => $this->academicScore($sid),
            'behavior_score' => $this->behaviorScore($sid),
            'attendance_rate' => $this->attendanceRate($sid),
        ])->all();

        $pyResponse = Http::withHeaders(['X-API-Key' => 'change-me-shared-secret'])
            ->timeout(30)
            ->post('http://localhost:8003/distribute', [
                'students'         => $students,
                'num_classes'      => $validated['num_classes'],
                'criteria_weights' => $validated['criteria_weights'],
            ]);

        if (! $pyResponse->successful()) {
            return response()->json([
                'message' => 'Distribution service error.',
                'detail'  => $pyResponse->body(),
            ], 502);
        }

        $result = $pyResponse->json();

        $distributionId = Str::uuid()->toString();

        ClassDistribution::create([
            'distribution_id'  => $distributionId,
            'class_id'         => $validated['class_id'],
            'num_classes'      => $validated['num_classes'],
            'criteria_weights' => $validated['criteria_weights'],
            'balance_score'    => $result['balance_metrics']['balance_score'] ?? null,
            'is_balanced'      => $result['balance_metrics']['is_balanced'] ?? null,
            'explanation'      => $result['balance_metrics']['explanation'] ?? null,
            'status'           => 'draft',
        ]);

        $rows = [];
        foreach ($result['classes'] as $className => $ids) {
            foreach ($ids as $sid) {
                $rows[] = [
                    'distribution_id' => $distributionId,
                    'student_id'      => (int) $sid,
                    'class_name'      => $className,
                    'created_at'      => now(),
                    'updated_at'      => now(),
                ];
            }
        }
        ClassAssignment::insert($rows);

        // Build a student_id → {name, email} map for the frontend to display
        $studentDetails = Student::with('user')
            ->whereIn('id', $studentIds)
            ->get()
            ->keyBy('id')
            ->map(fn ($s) => [
                'name'  => $s->user?->name ?? "Student #{$s->id}",
                'email' => $s->user?->email ?? '',
            ]);

        return response()->json([
            'distribution_id' => $distributionId,
            'classes'         => $result['classes'],
            'balance_metrics' => $result['balance_metrics'],
            'total_students'  => $result['total_students'],
            'num_classes'     => $result['num_classes'],
            'students_map'    => $studentDetails,
        ]);
    }

    public function confirm(string $distributionId)
    {
        $distribution = ClassDistribution::where('distribution_id', $distributionId)->firstOrFail();

        if ($distribution->status === 'confirmed') {
            return response()->json(['message' => 'Already confirmed.'], 409);
        }

        $distribution->update(['status' => 'confirmed']);

        return response()->json(['message' => 'Distribution confirmed and saved.']);
    }

    // ── Score helpers ────────────────────────────────────────────────────────

    private function academicScore(int $sid): float
    {
        $rows = AssessmentResult::where('student_id', $sid)
            ->join('assessment', 'assessmentresult.assessment_id', '=', 'assessment.assessment_id')
            ->selectRaw('assessmentresult.score, assessment.maxscore')
            ->get();

        if ($rows->isEmpty()) {
            return 50.0;
        }

        $percentages = $rows->filter(fn ($r) => $r->maxscore > 0)
            ->map(fn ($r) => ($r->score / $r->maxscore) * 100);

        return $percentages->isEmpty() ? 50.0 : round((float) $percentages->avg(), 2);
    }

    private function behaviorScore(int $sid): float
    {
        $logs = BehaviorLog::where('student_id', $sid)->get();

        if ($logs->isEmpty()) {
            return 75.0;
        }

        $score = 80.0;
        foreach ($logs as $log) {
            if ($log->type === 'positive') {
                $score += 5;
            } elseif ($log->type === 'negative') {
                $score -= 10;
            }
        }

        return (float) max(0, min(100, round($score, 2)));
    }

    private function attendanceRate(int $sid): float
    {
        $total = StudentAttendance::where('student_id', $sid)->count();

        if ($total === 0) {
            return 0.85;
        }

        $present = StudentAttendance::where('student_id', $sid)
            ->where('status', 'present')
            ->count();

        return round($present / $total, 4);
    }
}
