<?php

namespace App\Http\Controllers\Teacher;

use App\Http\Controllers\Controller;
use App\Models\Conversation;
use App\Models\Message;
use App\Models\Teacher;
use App\Models\User;
use Illuminate\Http\Request;

class ConversationController extends Controller
{
    /**
     * List all conversations for this teacher, most recent first.
     * Each item includes the other user's name, last message preview,
     * last message time, and unread count.
     */
    public function index(int $teacherId)
    {
        $teacher = Teacher::with('user')->findOrFail($teacherId);
        $userId  = $teacher->user_id;

        $conversations = Conversation::where('teacher_user_id', $userId)
            ->with(['parentUser', 'lastMessage'])
            ->orderByDesc('last_message_at')
            ->get()
            ->map(fn ($conv) => $this->_format($conv, $userId));

        return response()->json($conversations);
    }

    /**
     * Start a new conversation (or reuse an existing one) with a parent,
     * and send the first message.
     *
     * Body: { parent_user_id: int, body: string }
     */
    public function start(int $teacherId, Request $request)
    {
        $teacher = Teacher::with('user')->findOrFail($teacherId);
        $userId  = $teacher->user_id;

        $request->validate([
            'parent_user_id' => 'required|exists:users,id',
            'body'           => 'required|string|max:5000',
        ]);

        // Ensure the target is actually a parent.
        $parent = User::findOrFail($request->parent_user_id);
        if ($parent->role_type !== 'parent') {
            return response()->json(['message' => 'You can only message parents.'], 422);
        }

        $conversation = Conversation::firstOrCreate(
            ['teacher_user_id' => $userId, 'parent_user_id' => $parent->id]
        );

        $message = Message::create([
            'conversation_id' => $conversation->id,
            'sender_id'       => $userId,
            'receiver_id'     => $parent->id,
            'body'            => $request->body,
        ]);

        $conversation->update(['last_message_at' => $message->created_at]);

        return response()->json([
            'conversation_id' => $conversation->id,
            'message'         => $this->_msg($message),
        ], 201);
    }

    /**
     * Return all messages in a conversation (oldest first, paginated).
     * Accepts ?after_id=N to return only messages newer than N (for polling).
     * Marks all incoming messages as read automatically.
     */
    public function messages(int $teacherId, int $convId, Request $request)
    {
        $teacher = Teacher::with('user')->findOrFail($teacherId);
        $userId  = $teacher->user_id;

        $conv = Conversation::where('id', $convId)
            ->where('teacher_user_id', $userId)
            ->firstOrFail();

        // Mark incoming messages as read.
        Message::where('conversation_id', $conv->id)
            ->where('sender_id', '!=', $userId)
            ->whereNull('read_at')
            ->update(['read_at' => now()]);

        $query = Message::where('conversation_id', $conv->id)
            ->orderBy('id'); // oldest first

        if ($request->filled('after_id')) {
            $query->where('id', '>', (int) $request->after_id);
        }

        $perPage  = (int) $request->input('per_page', 50);
        $messages = $query->paginate($perPage);

        return response()->json([
            'data'         => $messages->map(fn ($m) => $this->_msg($m)),
            'total'        => $messages->total(),
            'per_page'     => $messages->perPage(),
            'current_page' => $messages->currentPage(),
            'last_page'    => $messages->lastPage(),
        ]);
    }

    /**
     * Send a reply message inside an existing conversation.
     *
     * Body: { body: string }
     */
    public function reply(int $teacherId, int $convId, Request $request)
    {
        $teacher = Teacher::with('user')->findOrFail($teacherId);
        $userId  = $teacher->user_id;

        $conv = Conversation::where('id', $convId)
            ->where('teacher_user_id', $userId)
            ->firstOrFail();

        $request->validate(['body' => 'required|string|max:5000']);

        $message = Message::create([
            'conversation_id' => $conv->id,
            'sender_id'       => $userId,
            'receiver_id'     => $conv->parent_user_id,
            'body'            => $request->body,
        ]);

        $conv->update(['last_message_at' => $message->created_at]);

        return response()->json($this->_msg($message), 201);
    }

    /**
     * Mark all incoming messages in a conversation as read.
     */
    public function markRead(int $teacherId, int $convId)
    {
        $teacher = Teacher::with('user')->findOrFail($teacherId);
        $userId  = $teacher->user_id;

        $conv = Conversation::where('id', $convId)
            ->where('teacher_user_id', $userId)
            ->firstOrFail();

        Message::where('conversation_id', $conv->id)
            ->where('sender_id', '!=', $userId)
            ->whereNull('read_at')
            ->update(['read_at' => now()]);

        return response()->json(['ok' => true]);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private function _format(Conversation $conv, int $myUserId): array
    {
        $otherUser   = $conv->parentUser;
        $last        = $conv->lastMessage;
        $unreadCount = Message::where('conversation_id', $conv->id)
            ->where('sender_id', '!=', $myUserId)
            ->whereNull('read_at')
            ->count();

        return [
            'id'               => $conv->id,
            'other_user_id'    => $otherUser?->id,
            'other_user_name'  => $otherUser?->name ?? '—',
            'last_message'     => $last?->body,
            'last_message_at'  => $last?->created_at?->toISOString(),
            'unread_count'     => $unreadCount,
        ];
    }

    private function _msg(Message $m): array
    {
        return [
            'id'         => $m->id,
            'sender_id'  => $m->sender_id,
            'body'       => $m->body,
            'created_at' => $m->created_at->toISOString(),
            'read_at'    => $m->read_at?->toISOString(),
        ];
    }
}
