<?php

namespace App\Http\Controllers\Teacher;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class QuestionGeneratorController extends Controller
{
    private const AI_BASE    = 'http://localhost:8002';
    private const AI_TIMEOUT = 3600; // AI generation can take 30–60 min on CPU

    private function aiKey(): string
    {
        return env('AI_API_KEY', 'change-me-shared-secret');
    }

    private function aiHttp()
    {
        return Http::withHeaders(['X-API-Key' => $this->aiKey()])
                   ->timeout(self::AI_TIMEOUT);
    }

    // ── GET /{teacherId}/question-generator/books ──────────────────────────────
    public function books(): \Illuminate\Http\JsonResponse
    {
        $response = $this->aiHttp()->get(self::AI_BASE . '/books');

        if ($response->failed()) {
            return response()->json(['message' => 'Could not reach question-generator service.'], 502);
        }

        return response()->json($response->json());
    }

    // ── GET /{teacherId}/question-generator/books/{book}/chapters ──────────────
    public function chapters(Request $request, int $teacherId, string $book): \Illuminate\Http\JsonResponse
    {
        $response = $this->aiHttp()->get(self::AI_BASE . '/books/' . urlencode($book) . '/chapters');

        if ($response->status() === 404) {
            return response()->json(['message' => "Book '{$book}' not found in curriculum index."], 404);
        }
        if ($response->failed()) {
            return response()->json(['message' => 'Could not reach question-generator service.'], 502);
        }

        return response()->json($response->json());
    }

    // ── POST /{teacherId}/question-generator/generate ──────────────────────────
    // Body: { document_id, chapters?, page_start?, page_end?, question_counts,
    //         bloom_levels, difficulty, language, evaluate }
    public function generate(Request $request, int $teacherId): \Illuminate\Http\JsonResponse
    {
        set_time_limit(0);

        $validated = $request->validate([
            'document_id'     => 'required|string',
            'chapters'        => 'nullable|array',
            'chapters.*'      => 'integer',
            'page_start'      => 'nullable|integer|min:1',
            'page_end'        => 'nullable|integer|min:1',
            'question_counts' => 'nullable|array',
            'bloom_levels'    => 'nullable|array',
            'bloom_levels.*'  => 'in:remember,understand,apply,analyze,evaluate,create',
            'difficulty'      => 'nullable|in:easy,medium,hard',
            'language'        => 'nullable|in:ar,en,auto',
            'topic_hint'      => 'nullable|string|max:500',
            'evaluate'        => 'nullable|boolean',
        ]);

        $payload = array_filter($validated, fn ($v) => $v !== null);

        $response = $this->aiHttp()->post(self::AI_BASE . '/questions/generate', $payload);

        if ($response->failed()) {
            $body   = $response->json();
            $status = in_array($response->status(), [400, 404, 422, 500, 503])
                ? $response->status() : 500;
            return response()->json([
                'message' => $body['detail'] ?? 'Question generation failed.',
            ], $status);
        }

        return response()->json($response->json());
    }

    // ── POST /{teacherId}/question-generator/export ────────────────────────────
    // Body: { questions, title, language, include_answers }
    // Returns: binary .docx
    public function export(Request $request, int $teacherId)
    {
        $validated = $request->validate([
            'questions'       => 'required|array',
            'title'           => 'nullable|string|max:255',
            'document_id'     => 'nullable|string',
            'difficulty'      => 'nullable|string',
            'language'        => 'nullable|in:ar,en,auto',
            'include_answers' => 'nullable|boolean',
        ]);

        $payload = array_filter($validated, fn ($v) => $v !== null);

        $response = $this->aiHttp()->post(self::AI_BASE . '/questions/export/docx', $payload);

        if ($response->failed()) {
            $body   = $response->json();
            $status = in_array($response->status(), [400, 422, 500]) ? $response->status() : 500;
            return response()->json([
                'message' => $body['detail'] ?? 'Word export failed.',
            ], $status);
        }

        $filename = 'exam_' . now()->format('Ymd_His') . '.docx';

        return response($response->body(), 200, [
            'Content-Type'        => 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'Content-Disposition' => 'attachment; filename="' . $filename . '"',
        ]);
    }

    // ── POST /{teacherId}/generate-exam  (legacy / PDF upload path) ───────────
    // Kept for non-curriculum PDFs. Calls /questions/from-pdf on the AI service.
    public function fromPdf(Request $request, int $teacherId)
    {
        set_time_limit(0);
        $request->validate([
            'pdf'                => 'required|file|mimes:pdf|max:409600',
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

        $response = Http::withHeaders(['X-API-Key' => $this->aiKey()])
            ->timeout(self::AI_TIMEOUT)
            ->attach('file', file_get_contents($pdf->path()), $pdf->getClientOriginalName())
            ->post(self::AI_BASE . '/questions/from-pdf', $data);

        if ($response->failed()) {
            $body   = $response->json();
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
