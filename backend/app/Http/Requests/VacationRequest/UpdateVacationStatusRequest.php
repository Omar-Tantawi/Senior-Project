<?php

namespace App\Http\Requests\VacationRequest;

use Illuminate\Foundation\Http\FormRequest;

class UpdateVacationStatusRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'status' => 'required|in:approved,rejected',
        ];
    }
}
