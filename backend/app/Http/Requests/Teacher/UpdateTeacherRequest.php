<?php

namespace App\Http\Requests\Teacher;

use App\Models\Teacher;
use Illuminate\Foundation\Http\FormRequest;

class UpdateTeacherRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        $userId = Teacher::findOrFail($this->route('id'))->user_id;

        return [
            'name'          => 'sometimes|string|max:255',
            'email'         => "sometimes|email|unique:users,email,{$userId}",
            'phone'         => 'nullable|string|max:20',
            'date_of_birth' => 'nullable|date',
            'gender'        => 'nullable|in:male,female,other',
            'address'       => 'nullable|string',
            'hire_date'     => 'nullable|date',
            'status'        => 'nullable|in:active,inactive,resigned',
            'is_active'     => 'nullable|boolean',
        ];
    }
}
