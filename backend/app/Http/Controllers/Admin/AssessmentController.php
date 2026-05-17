<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Assessment\StoreAssessmentRequest;
use App\Http\Requests\Assessment\UpdateAssessmentRequest;
use App\Models\Assessment;
use App\Models\AssessmentResult;
use App\Models\Enrollment;
use App\Repositories\AssessmentRepository;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class AssessmentController extends Controller
{
    public function __construct(private AssessmentRepository $repo) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filter(
                $request->only(['subject_id', 'section_id', 'assessmenttype']),
                $request->input('per_page', 15)
            )
        );
    }

    public function store(StoreAssessmentRequest $request)
    {
        $assessment = Assessment::create($request->validated());
        return response()->json($assessment->load(['subject', 'section.schoolClass']), 201);
    }

    public function show(int $id)
    {
        return response()->json(
            Assessment::with(['subject', 'section.schoolClass.schoolYear', 'results.student.user'])->findOrFail($id)
        );
    }

    public function storeResults(Request $request, int $id)
    {
        $assessment = Assessment::findOrFail($id);

        $request->validate([
            'results'              => 'required|array|min:1',
            'results.*.student_id' => 'required|exists:students,id',
            'results.*.score'      => "required|numeric|min:0|max:{$assessment->maxscore}",
        ]);

        $enrolledIds = Enrollment::where('section_id', $assessment->section_id)
            ->where('status', 'active')
            ->pluck('student_id')
            ->toArray();

        $invalidStudents = collect($request->results)->pluck('student_id')->diff($enrolledIds);

        if ($invalidStudents->isNotEmpty()) {
            return response()->json([
                'message'          => 'Some students are not enrolled in this section.',
                'invalid_students' => $invalidStudents->values(),
            ], 422);
        }

        $now  = now();
        $rows = collect($request->results)->map(fn($item) => [
            'assessment_id' => $assessment->assessment_id,
            'student_id'    => $item['student_id'],
            'score'         => $item['score'],
            'grade'         => AssessmentResult::calculateGrade($item['score'], $assessment->maxscore),
            'publishedat'   => $now,
        ])->toArray();

        DB::transaction(fn() => AssessmentResult::upsert(
            $rows,
            ['assessment_id', 'student_id'],
            ['score', 'grade', 'publishedat']
        ));

        return response()->json(['message' => 'Marks saved successfully.', 'count' => count($rows)]);
    }

    public function results(int $id)
    {
        $assessment = Assessment::with(['subject', 'section.schoolClass'])->findOrFail($id);

        $results = AssessmentResult::with('student.user')
            ->where('assessment_id', $id)
            ->get()
            ->map(fn($r) => [
                'result_id'    => $r->result_id,
                'student_id'   => $r->student_id,
                'student_name' => $r->student?->user?->name ?? '—',
                'score'        => $r->score,
                'max_score'    => $assessment->maxscore,
                'percentage'   => $assessment->maxscore > 0
                    ? round(($r->score / $assessment->maxscore) * 100, 1) : 0,
                'grade'        => $r->grade,
                'published_at' => $r->publishedat,
            ]);

        $enrolledCount = DB::table('enrollment')
            ->where('section_id', $assessment->section_id)
            ->where('status', 'active')
            ->count();

        return response()->json([
            'assessment' => [
                'id'             => $assessment->assessment_id,
                'title'          => $assessment->title,
                'subject'        => $assessment->subject?->name ?? '—',
                'section'        => $assessment->section?->name ?? '—',
                'class'          => $assessment->section?->schoolClass?->name ?? '—',
                'assessmenttype' => $assessment->assessmenttype,
                'date'           => $assessment->date,
                'maxscore'       => $assessment->maxscore,
            ],
            'results' => $results,
            'summary' => [
                'submitted'     => $results->count(),
                'enrolled'      => $enrolledCount,
                'average_score' => round($results->avg('score'), 2),
                'highest_score' => $results->max('score'),
                'lowest_score'  => $results->min('score'),
                'pass_rate'     => $results->count()
                    ? round(($results->where('grade', '!=', 'F')->count() / $results->count()) * 100, 1) : 0,
            ],
        ]);
    }

    public function update(UpdateAssessmentRequest $request, int $id)
    {
        $assessment = Assessment::findOrFail($id);
        $assessment->update($request->validated());
        return response()->json($assessment->load(['subject', 'section.schoolClass']));
    }

    public function destroy(int $id)
    {
        Assessment::findOrFail($id)->delete();
        return response()->json(['message' => 'Assessment deleted successfully.']);
    }
}
