<?php

namespace App\Http\Controllers\Teacher;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class QuestionGeneratorController extends Controller
{
    public function generate(Request $request, int $teacherId)
    {
        // AI generation can take 30-60 min on CPU — disable PHP's execution limit.
        set_time_limit(0);
        $request->validate([
            'pdf'                => 'required|file|mimes:pdf|max:409600', // 400 MB
            'page_start'         => 'nullable|integer|min:1',
            'page_end'           => 'nullable|integer|min:1',
            'mcq_count'          => 'nullable|integer|min:0|max:50',
            'true_false_count'   => 'nullable|integer|min:0|max:50',
            'short_answer_count' => 'nullable|integer|min:0|max:50',
            'fill_blank_count'   => 'nullable|integer|min:0|max:50',
            'essay_count'        => 'nullable|integer|min:0|max:20',
            'difficulty'         => 'nullable|in:easy,medium,hard',
            'language'           => 'nullable|in:ar,en,auto',
            'title'              => 'nullable|string|max:255',
            'bloom_levels'       => 'nullable|string',
            'include_answers'    => 'nullable|in:true,false',
            'output'             => 'nullable|in:docx,json',
        ]);

        $pdf  = $request->file('pdf');
        $data = array_filter([
            'page_start'         => $request->page_start,
            'page_end'           => $request->page_end,
            'mcq_count'          => $request->input('mcq_count', 5),
            'true_false_count'   => $request->input('true_false_count', 3),
            'short_answer_count' => $request->input('short_answer_count', 2),
            'fill_blank_count'   => $request->input('fill_blank_count', 3),
            'essay_count'        => $request->input('essay_count', 0),
            'difficulty'         => $request->input('difficulty', 'medium'),
            'language'           => $request->input('language', 'auto'),
            'title'              => $request->input('title', 'Exam Questions'),
            'bloom_levels'       => $request->input('bloom_levels', 'remember,understand,apply'),
            'include_answers'    => $request->input('include_answers', 'true'),
            'output'             => $request->input('output', 'docx'),
        ], fn ($v) => $v !== null && $v !== '');

        $response = Http::withHeaders([
                'X-API-Key' => env('AI_API_KEY', 'change-me-shared-secret'),
            ])
            ->timeout(3600)          // AI generation can take 30–60 min on CPU
            ->attach(
                'file',
                file_get_contents($pdf->path()),
                $pdf->getClientOriginalName()
            )
            ->post('http://localhost:8002/questions/from-pdf', $data);

        if ($response->failed()) {
            $body   = $response->json();
            // Never forward 401 from the Python service — it would look like a
            // Sanctum auth failure to the client and log the teacher out.
            $status = in_array($response->status(), [400, 422, 500, 503]) ? $response->status() : 500;
            return response()->json([
                'message' => $body['detail'] ?? 'Question generation failed.',
            ], $status);
        }

        $filename = 'exam_' . now()->format('Ymd_His') . '.docx';

        return response($response->body(), 200, [
            'Content-Type'        => 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'Content-Disposition' => 'attachment; filename="' . $filename . '"',
        ]);
    }
}
