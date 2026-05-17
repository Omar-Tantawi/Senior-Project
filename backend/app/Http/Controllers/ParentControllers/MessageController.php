<?php

namespace App\Http\Controllers\ParentControllers;

use App\Http\Controllers\Controller;
use App\Http\Requests\Message\SendMessageRequest;
use App\Models\Guardian;
use App\Models\Message;
use App\Models\Teacher;
use Illuminate\Http\Request;

class MessageController extends Controller
{
    public function index(int $parentId, Request $request)
    {
        $guardian = Guardian::findOrFail($parentId);

        $messages = Message::where('sender_id', $guardian->user_id)
            ->orWhere('receiver_id', $guardian->user_id)
            ->with(['sender', 'receiver', 'student.user'])
            ->orderByDesc('created_at')
            ->paginate($request->input('per_page', 20));

        return response()->json($messages);
    }

    public function send(int $parentId, SendMessageRequest $request)
    {
        $guardian = Guardian::findOrFail($parentId);
        $data     = $request->validated();
        $teacher  = Teacher::with('user')->findOrFail($data['teacher_id']);

        $message = Message::create([
            'sender_id'   => $guardian->user_id,
            'receiver_id' => $teacher->user_id,
            'student_id'  => $data['student_id'] ?? null,
            'subject'     => $data['subject'],
            'body'        => $data['body'],
        ]);

        return response()->json($message->load(['sender', 'receiver', 'student.user']), 201);
    }

    public function show(int $parentId, int $id)
    {
        $guardian = Guardian::findOrFail($parentId);

        $message = Message::where('id', $id)
            ->where(fn ($q) => $q->where('sender_id', $guardian->user_id)->orWhere('receiver_id', $guardian->user_id))
            ->with(['sender', 'receiver', 'student.user'])
            ->firstOrFail();

        if ($message->receiver_id === $guardian->user_id && ! $message->read_at) {
            $message->update(['read_at' => now()]);
        }

        return response()->json($message);
    }
}
