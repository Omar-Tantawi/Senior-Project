<?php

namespace App\Http\Requests\Parent;

use Illuminate\Foundation\Http\FormRequest;

class AddChildRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'student_id'   => 'required|exists:students,id',
            'relationship' => 'required|string|max:50',
            'isprimary'    => 'sometimes|boolean',
        ];
    }
}
