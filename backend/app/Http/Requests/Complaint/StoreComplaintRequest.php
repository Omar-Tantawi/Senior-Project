<?php

namespace App\Http\Requests\Complaint;

use Illuminate\Foundation\Http\FormRequest;

class StoreComplaintRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'student_id' => 'nullable|integer',
            'subject'    => 'required|string|max:255',
            'body'       => 'required|string',
        ];
    }
}
